cmake_minimum_required(VERSION 3.15)
project(KarplusStrongPlugin VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile commands for IDE support
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# =============================================================================
# Options
# =============================================================================
option(BUILD_TESTS "Build test targets" OFF)
option(USE_SANITIZERS "Enable address/undefined sanitizers (Debug only)" OFF)

# =============================================================================
# JUCE Dependency
# =============================================================================
# Priority: 1) Local JUCE folder, 2) JUCE_DIR env var, 3) FetchContent
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/JUCE/CMakeLists.txt")
    message(STATUS "Using local JUCE from ${CMAKE_CURRENT_SOURCE_DIR}/JUCE")
    add_subdirectory(JUCE)
elseif(DEFINED ENV{JUCE_DIR})
    message(STATUS "Using JUCE from JUCE_DIR: $ENV{JUCE_DIR}")
    add_subdirectory($ENV{JUCE_DIR} JUCE)
elseif(DEFINED JUCE_DIR)
    message(STATUS "Using JUCE from JUCE_DIR: ${JUCE_DIR}")
    add_subdirectory(${JUCE_DIR} JUCE)
else()
    message(STATUS "Fetching JUCE via FetchContent...")
    include(FetchContent)
    FetchContent_Declare(
        JUCE
        GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
        GIT_TAG 7.0.9
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(JUCE)
endif()

# =============================================================================
# Platform-specific plugin formats
# =============================================================================
set(PLUGIN_FORMATS VST3 Standalone)

if(APPLE)
    list(APPEND PLUGIN_FORMATS AU)
endif()

# AAX requires separate SDK license from Avid
# LV2 can be added if needed: list(APPEND PLUGIN_FORMATS LV2)

# =============================================================================
# Plugin Target
# =============================================================================
juce_add_plugin(KarplusStrongPlugin
    COMPANY_NAME "UNIX Audio"
    COMPANY_WEBSITE "https://github.com/your-username/karplus-strong"
    COMPANY_EMAIL "contact@example.com"

    PLUGIN_MANUFACTURER_CODE Unix
    PLUGIN_CODE Kpls

    IS_SYNTH TRUE
    NEEDS_MIDI_INPUT TRUE
    NEEDS_MIDI_OUTPUT FALSE
    IS_MIDI_EFFECT FALSE
    EDITOR_WANTS_KEYBOARD_FOCUS TRUE

    COPY_PLUGIN_AFTER_BUILD FALSE

    FORMATS ${PLUGIN_FORMATS}
    PRODUCT_NAME "Karplus-Strong Synth"

    # macOS specific
    BUNDLE_ID "com.unixaudio.karplusstrong"
    HARDENED_RUNTIME_ENABLED TRUE

    # VST3 category
    VST3_CATEGORIES Instrument Synth
)

# =============================================================================
# Source Files
# =============================================================================
target_sources(KarplusStrongPlugin
    PRIVATE
        Source/PluginProcessor.cpp
        Source/PluginEditor.cpp
        Source/Components/WaveformDisplay.cpp
        Source/Components/StringVisualization.cpp
        Source/Components/VirtualKeyboard.cpp
)

target_include_directories(KarplusStrongPlugin
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/DSP
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/Components
)

# =============================================================================
# Compile Definitions
# =============================================================================
target_compile_definitions(KarplusStrongPlugin
    PUBLIC
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
        JUCE_VST3_CAN_REPLACE_VST2=0
        JUCE_DISPLAY_SPLASH_SCREEN=0
        JUCE_REPORT_APP_USAGE=0
        # Silence JUCE deprecation warnings
        JUCE_MODAL_LOOPS_PERMITTED=1
)

# =============================================================================
# Platform-specific settings
# =============================================================================
if(MSVC)
    # Windows/MSVC settings
    target_compile_options(KarplusStrongPlugin PRIVATE
        /W4             # Warning level 4
        /permissive-    # Strict conformance
        /Zc:__cplusplus # Correct __cplusplus macro
        /MP             # Multi-processor compilation
    )
    target_compile_definitions(KarplusStrongPlugin PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX        # Prevent min/max macro conflicts
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    # macOS/Linux with Clang or GCC
    target_compile_options(KarplusStrongPlugin PRIVATE
        -Wall
        -Wextra
        -Wpedantic
        -Wno-unused-parameter
    )

    if(USE_SANITIZERS AND CMAKE_BUILD_TYPE STREQUAL "Debug")
        target_compile_options(KarplusStrongPlugin PRIVATE
            -fsanitize=address,undefined
            -fno-omit-frame-pointer
        )
        target_link_options(KarplusStrongPlugin PRIVATE
            -fsanitize=address,undefined
        )
    endif()
endif()

# Linux-specific dependencies
if(UNIX AND NOT APPLE)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(ALSA REQUIRED alsa)
    pkg_check_modules(FREETYPE2 REQUIRED freetype2)
    pkg_check_modules(WEBKIT2GTK webkit2gtk-4.0)  # Optional

    target_link_libraries(KarplusStrongPlugin
        PRIVATE
            ${ALSA_LIBRARIES}
            ${FREETYPE2_LIBRARIES}
            pthread
            dl
    )

    # X11 dependencies for JUCE GUI
    find_package(X11 REQUIRED)
    target_link_libraries(KarplusStrongPlugin PRIVATE ${X11_LIBRARIES})

    if(X11_Xrandr_FOUND)
        target_link_libraries(KarplusStrongPlugin PRIVATE ${X11_Xrandr_LIB})
    endif()
    if(X11_Xinerama_FOUND)
        target_link_libraries(KarplusStrongPlugin PRIVATE ${X11_Xinerama_LIB})
    endif()
    if(X11_Xcursor_FOUND)
        target_link_libraries(KarplusStrongPlugin PRIVATE ${X11_Xcursor_LIB})
    endif()
endif()

# =============================================================================
# Link Libraries
# =============================================================================
target_link_libraries(KarplusStrongPlugin
    PRIVATE
        juce::juce_audio_utils
        juce::juce_audio_processors
    PUBLIC
        juce::juce_recommended_config_flags
        juce::juce_recommended_lto_flags
        juce::juce_recommended_warning_flags
)

# =============================================================================
# Tests (Optional)
# =============================================================================
if(BUILD_TESTS)
    enable_testing()

    # Unit test executable using JUCE's test framework
    juce_add_console_app(KarplusStrongTests
        PRODUCT_NAME "Karplus-Strong Tests"
    )

    target_sources(KarplusStrongTests PRIVATE
        Tests/DSPTests.cpp
    )

    target_include_directories(KarplusStrongTests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/Source
        ${CMAKE_CURRENT_SOURCE_DIR}/Source/DSP
    )

    target_link_libraries(KarplusStrongTests PRIVATE
        juce::juce_audio_basics
        juce::juce_core
    )

    target_compile_definitions(KarplusStrongTests PRIVATE
        JUCE_WEB_BROWSER=0
        JUCE_USE_CURL=0
    )

    add_test(NAME DSPTests COMMAND KarplusStrongTests)
endif()

# =============================================================================
# Installation
# =============================================================================
# Install VST3 to standard locations
if(APPLE)
    set(VST3_INSTALL_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/VST3")
    set(AU_INSTALL_DIR "$ENV{HOME}/Library/Audio/Plug-Ins/Components")
elseif(UNIX)
    set(VST3_INSTALL_DIR "$ENV{HOME}/.vst3")
elseif(WIN32)
    set(VST3_INSTALL_DIR "$ENV{COMMONPROGRAMFILES}/VST3")
endif()

# Custom install target
add_custom_target(install_plugin
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "$<TARGET_FILE_DIR:KarplusStrongPlugin_VST3>/../"
        "${VST3_INSTALL_DIR}/Karplus-Strong Synth.vst3"
    DEPENDS KarplusStrongPlugin_VST3
    COMMENT "Installing VST3 to ${VST3_INSTALL_DIR}"
)

# =============================================================================
# Pluginval Integration (for CI testing)
# =============================================================================
# Download pluginval for automated plugin validation
# Usage: cmake --build . --target run_pluginval
find_program(PLUGINVAL_EXECUTABLE pluginval
    HINTS
        "${CMAKE_CURRENT_SOURCE_DIR}/tools"
        "$ENV{HOME}/bin"
        "/usr/local/bin"
)

if(PLUGINVAL_EXECUTABLE)
    add_custom_target(run_pluginval
        COMMAND ${PLUGINVAL_EXECUTABLE}
            --strictness-level 5
            --validate-in-process
            --timeout-ms 60000
            "$<TARGET_FILE:KarplusStrongPlugin_VST3>"
        DEPENDS KarplusStrongPlugin_VST3
        COMMENT "Running pluginval validation..."
    )
else()
    message(STATUS "pluginval not found - validation target disabled")
    message(STATUS "Download from: https://github.com/Tracktion/pluginval/releases")
endif()

# =============================================================================
# Summary
# =============================================================================
message(STATUS "")
message(STATUS "Karplus-Strong Plugin Configuration:")
message(STATUS "  Version:    ${PROJECT_VERSION}")
message(STATUS "  Formats:    ${PLUGIN_FORMATS}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Platform:   ${CMAKE_SYSTEM_NAME}")
message(STATUS "")
