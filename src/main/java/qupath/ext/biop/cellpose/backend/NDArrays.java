/*-
 * Copyright 2026 QuPath developers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.biop.cellpose.backend;

import ij.ImagePlus;
import ij.ImageStack;
import ij.process.ByteProcessor;
import ij.process.FloatProcessor;
import ij.process.ImageProcessor;
import ij.process.ShortProcessor;
import org.apposed.appose.NDArray;
import org.apposed.appose.NDArray.DType;
import org.apposed.appose.NDArray.Shape;
import org.apposed.appose.NDArray.Shape.Order;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Marshalling between ImageJ image data and Appose {@link NDArray} shared-memory buffers.
 * <p>
 * This is Apache-2.0 original work, written independently against the public Appose
 * {@link NDArray} API. It does not reuse any ImgLib2/imglib2-appose helper. The conventions are:
 * <ul>
 *     <li>the DType is chosen from the ImageJ processor bit depth: 8-bit -&gt; UINT8,
 *     16-bit -&gt; UINT16, 32-bit float -&gt; FLOAT32;</li>
 *     <li>arrays are laid out in C order (row-major) with the shape (height, width) for a single
 *     plane, or (channels, height, width) for a multi-channel image;</li>
 *     <li>pixels are copied through {@link NDArray#buffer()} honouring the native byte order, which
 *     is what the numpy view on the Python side expects.</li>
 * </ul>
 * Every {@link NDArray} allocated here owns a shared-memory segment and must be
 * {@link NDArray#close() closed} by the caller once done, to avoid leaking shared memory.
 */
final class NDArrays {

    private NDArrays() {}

    /**
     * Choose the Appose DType matching an ImageJ processor.
     *
     * @param ip the processor
     * @return the matching {@link DType}
     * @throws IllegalArgumentException if the processor type is unsupported (e.g. RGB/ColorProcessor)
     */
    static DType dTypeFor(ImageProcessor ip) {
        if (ip instanceof ByteProcessor)
            return DType.UINT8;
        if (ip instanceof ShortProcessor)
            return DType.UINT16;
        if (ip instanceof FloatProcessor)
            return DType.FLOAT32;
        throw new IllegalArgumentException("Unsupported ImageProcessor type for Appose marshalling: "
                + ip.getClass().getSimpleName() + " (expected 8-bit, 16-bit or 32-bit float)");
    }

    /**
     * Convert an ImageJ {@link ImagePlus} to an input {@link NDArray}. Each slice of the stack is
     * treated as one channel. The resulting shape is (height, width) for a single-channel image,
     * or (channels, height, width) otherwise, in C order.
     *
     * @param imp the image; all slices must share the same processor type
     * @return a freshly allocated NDArray; the caller must close it
     */
    static NDArray fromImagePlus(ImagePlus imp) {
        int width = imp.getWidth();
        int height = imp.getHeight();
        ImageStack stack = imp.getStack();
        int nChannels = stack.getSize();

        DType dType = dTypeFor(stack.getProcessor(1));
        Shape shape = nChannels == 1
                ? new Shape(Order.C_ORDER, height, width)
                : new Shape(Order.C_ORDER, nChannels, height, width);

        NDArray ndArray = new NDArray(dType, shape);
        ByteBuffer buffer = ndArray.buffer().order(ByteOrder.nativeOrder());
        buffer.rewind();
        for (int c = 1; c <= nChannels; c++) {
            writeProcessor(stack.getProcessor(c), buffer, dType);
        }
        return ndArray;
    }

    /**
     * Convert selected slices of an {@link ImagePlus} to an input {@link NDArray}, packed compactly:
     * shape (height, width) for a single band, or (bands, height, width) otherwise, in C order.
     * <p>
     * Packing the chosen channels compactly -- rather than sending the whole stack and asking
     * Cellpose to index into it -- keeps the array within the channel count Cellpose can interpret.
     *
     * @param imp   the image
     * @param bands 0-based slice indices to include, in the order Cellpose should see them
     * @return a freshly allocated NDArray; the caller must close it
     */
    static NDArray fromImagePlusChannels(ImagePlus imp, int[] bands) {
        int width = imp.getWidth();
        int height = imp.getHeight();
        ImageStack stack = imp.getStack();

        DType dType = dTypeFor(stack.getProcessor(1));
        Shape shape = bands.length == 1
                ? new Shape(Order.C_ORDER, height, width)
                : new Shape(Order.C_ORDER, bands.length, height, width);

        NDArray ndArray = new NDArray(dType, shape);
        ByteBuffer buffer = ndArray.buffer().order(ByteOrder.nativeOrder());
        buffer.rewind();
        for (int band : bands) {
            writeProcessor(stack.getProcessor(band + 1), buffer, dType);
        }
        return ndArray;
    }

    /**
     * Average every slice of an {@link ImagePlus} into a single FLOAT32 plane of shape
     * (height, width) in C order.
     * <p>
     * This reproduces, on the Java side, what Cellpose does for the grayscale channel spec
     * {@code [0, 0]} (it averages the channel axis). Doing it here means Cellpose is handed a plain
     * 2D plane: above three channels it otherwise mistakes the channel axis for a Z axis -- logging
     * {@code "z_axis not specified, assuming it is dim 0"} and returning an empty mask -- even when
     * {@code channel_axis} is passed explicitly. FLOAT32 keeps the mean exact for integer inputs.
     *
     * @param imp the image
     * @return a freshly allocated NDArray; the caller must close it
     */
    static NDArray meanOfChannels(ImagePlus imp) {
        int width = imp.getWidth();
        int height = imp.getHeight();
        ImageStack stack = imp.getStack();
        int nChannels = stack.getSize();

        NDArray ndArray = new NDArray(DType.FLOAT32, new Shape(Order.C_ORDER, height, width));
        ByteBuffer buffer = ndArray.buffer().order(ByteOrder.nativeOrder());
        buffer.rewind();
        ImageProcessor[] processors = new ImageProcessor[nChannels];
        for (int c = 0; c < nChannels; c++)
            processors[c] = stack.getProcessor(c + 1);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                double sum = 0;
                for (ImageProcessor ip : processors)
                    sum += ip.getf(x, y);
                buffer.putFloat((float) (sum / nChannels));
            }
        }
        return ndArray;
    }

    /**
     * Convert a single ImageJ {@link ImageProcessor} to a 2D input {@link NDArray} of shape
     * (height, width) in C order.
     *
     * @param ip the processor
     * @return a freshly allocated NDArray; the caller must close it
     */
    static NDArray fromProcessor(ImageProcessor ip) {
        DType dType = dTypeFor(ip);
        Shape shape = new Shape(Order.C_ORDER, ip.getHeight(), ip.getWidth());
        NDArray ndArray = new NDArray(dType, shape);
        ByteBuffer buffer = ndArray.buffer().order(ByteOrder.nativeOrder());
        buffer.rewind();
        writeProcessor(ip, buffer, dType);
        return ndArray;
    }

    /**
     * Allocate a 2D label {@link NDArray} of shape (height, width) in C order, using UINT16, which
     * matches the label output written back by the Cellpose scripts (up to 65535 labels per tile).
     *
     * @param width  the label image width
     * @param height the label image height
     * @return a freshly allocated NDArray; the caller must close it
     */
    static NDArray allocateLabels(int width, int height) {
        Shape shape = new Shape(Order.C_ORDER, height, width);
        return new NDArray(DType.UINT16, shape);
    }

    /**
     * Read a 2D UINT16 label {@link NDArray} back into an ImageJ {@link ShortProcessor}.
     *
     * @param labels a UINT16 NDArray with shape (height, width) in C order
     * @param width  the expected width
     * @param height the expected height
     * @return a new ShortProcessor holding the label values
     */
    static ShortProcessor labelsToShortProcessor(NDArray labels, int width, int height) {
        ByteBuffer buffer = labels.buffer().order(ByteOrder.nativeOrder());
        buffer.rewind();
        short[] pixels = new short[width * height];
        for (int i = 0; i < pixels.length; i++) {
            pixels[i] = buffer.getShort();
        }
        return new ShortProcessor(width, height, pixels, null);
    }

    /**
     * Read a 2D {@link NDArray} back into an ImageJ {@link ImageProcessor} of the matching type.
     * Supports UINT8, UINT16 and FLOAT32 with shape (height, width) in C order. Intended primarily
     * for round-trip verification.
     *
     * @param ndArray the NDArray to convert
     * @return a new ImageProcessor of the matching type
     */
    static ImageProcessor toProcessor(NDArray ndArray) {
        Shape shape = ndArray.shape();
        if (shape.length() != 2)
            throw new IllegalArgumentException("Expected a 2D NDArray, got " + shape.length() + " dimensions");
        int height = shape.get(0);
        int width = shape.get(1);
        ByteBuffer buffer = ndArray.buffer().order(ByteOrder.nativeOrder());
        buffer.rewind();
        int n = width * height;
        switch (ndArray.dType()) {
            case UINT8: {
                byte[] pixels = new byte[n];
                buffer.get(pixels);
                return new ByteProcessor(width, height, pixels, null);
            }
            case UINT16: {
                short[] pixels = new short[n];
                for (int i = 0; i < n; i++)
                    pixels[i] = buffer.getShort();
                return new ShortProcessor(width, height, pixels, null);
            }
            case FLOAT32: {
                float[] pixels = new float[n];
                for (int i = 0; i < n; i++)
                    pixels[i] = buffer.getFloat();
                return new FloatProcessor(width, height, pixels, null);
            }
            default:
                throw new IllegalArgumentException("Unsupported DType for conversion back to ImageProcessor: "
                        + ndArray.dType());
        }
    }

    private static void writeProcessor(ImageProcessor ip, ByteBuffer buffer, DType dType) {
        int width = ip.getWidth();
        int height = ip.getHeight();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                switch (dType) {
                    case UINT8:
                        buffer.put((byte) (ip.get(x, y) & 0xff));
                        break;
                    case UINT16:
                        buffer.putShort((short) (ip.get(x, y) & 0xffff));
                        break;
                    case FLOAT32:
                        buffer.putFloat(ip.getf(x, y));
                        break;
                    default:
                        throw new IllegalArgumentException("Unsupported DType for marshalling: " + dType);
                }
            }
        }
    }
}
