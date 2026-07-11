plugins {
    id("maven-publish")
    // QuPath Gradle extension convention plugin
    id("qupath-conventions")
}

qupathExtension {
    name = "qupath-extension-cellpose"
    group = "io.github.qupath"
    version = "0.12.2-SNAPSHOT"
    description = "QuPath extension to use Cellpose"
    automaticModule = "qupath.ext.biop.cellpose"
}

dependencies {
    implementation(libs.qupath.gui.fx)
    implementation(libs.qupath.fxtras)
    implementation(libs.extensionmanager)
    implementation("commons-io:commons-io:2.15.0")

    // In-process Python transport (opt-in Appose backend). Appose uses Groovy only for its
    // Java-side worker-protocol JSON. QuPath 0.7 already bundles Groovy core (5.0.4), so we exclude
    // Appose's Groovy to avoid a second, differing core (Appose pins Groovy 4.x). QuPath does NOT
    // ship the groovy-json module, though, and Appose needs groovy.json.JsonOutput -- so we add just
    // that module, pinned to QuPath's Groovy version, giving exactly one compatible Groovy on the
    // runtime classpath. JNA and jna-platform are kept: they back the NDArray shared-memory
    // transport. Appose is inert unless the Appose transport is chosen. See `./gradlew dependencies`.
    implementation("org.apposed:appose:0.12.0") {
        exclude(group = "org.apache.groovy")
    }
    implementation("org.apache.groovy:groovy-json:5.0.4")

    testImplementation(platform("org.junit:junit-bom:5.11.3"))
    testImplementation("org.junit.jupiter:junit-jupiter")
    testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

tasks.withType<Test> {
    useJUnitPlatform()
}

/*
 * Set HTML language and destination folder
 */
tasks.withType<Javadoc> {
    (options as StandardJavadocDocletOptions).addBooleanOption("html5", true)
    setDestinationDir(File(project.rootDir,"docs"))
}

/*
 * Avoid "Entry .gitkeep is a duplicate but no duplicate handling strategy has been set."
 * when using withSourcesJar()
 */
tasks.withType<org.gradle.jvm.tasks.Jar> {
    duplicatesStrategy = DuplicatesStrategy.INCLUDE
}

publishing {
    repositories {
        maven {
            name = "SciJava"
            val releasesRepoUrl = "https://maven.scijava.org/content/repositories/releases"
            val snapshotsRepoUrl = "https://maven.scijava.org/content/repositories/snapshots"
            url = if (project.hasProperty("release")) uri(releasesRepoUrl) else uri(snapshotsRepoUrl)
            credentials {
                username = System.getenv("MAVEN_USER")
                password = System.getenv("MAVEN_PASS")
            }
        }
    }

    publications {
        create<MavenPublication>("mavenJava") {
            from(components["java"])
            pom {
                licenses {
                    license {
                        name = "Apache License v2.0"
                        url = "http://www.apache.org/licenses/LICENSE-2.0"
                    }
                }
            }
        }
    }
}