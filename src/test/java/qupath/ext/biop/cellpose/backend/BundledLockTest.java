package qupath.ext.biop.cellpose.backend;

import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Checks on the committed pixi lock that a regenerated lock could silently undo. */
class BundledLockTest {

    private static final Pattern LINUX_OPENCV = Pattern.compile(
            "opencv_python_headless-(\\d+)\\.(\\d+)\\.[^/\\s]*manylinux[^/\\s]*\\.whl");

    /**
     * opencv-python-headless 4.13 and later bundle a FIPS build of OpenSSL whose self-test aborts
     * {@code import cv2} on Linux hosts in FIPS mode (opencv/opencv-python#1191), so the Linux
     * environments must stay on 4.12 until a fixed wheel exists.
     */
    @Test
    void linuxEnvironmentsLockAnOpencvWithoutTheFipsOpenssl() throws IOException {
        String lock = ApposeEnvironments.readResource("pixi.lock");
        Matcher m = LINUX_OPENCV.matcher(lock);
        List<String> found = new java.util.ArrayList<>();
        while (m.find()) {
            int major = Integer.parseInt(m.group(1));
            int minor = Integer.parseInt(m.group(2));
            found.add(m.group());
            assertTrue(major < 4 || (major == 4 && minor <= 12),
                    "The lock installs " + m.group() + " on Linux; 4.13+ aborts on FIPS hosts");
        }
        assertFalse(found.isEmpty(), "no Linux opencv-python-headless wheel found in the lock");
    }
}
