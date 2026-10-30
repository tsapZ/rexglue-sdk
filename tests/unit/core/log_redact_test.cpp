/**
 * @file        core/log_redact_test.cpp
 * @brief       Tests for RedactUserData (the player's data out of log lines)
 *
 * @license     BSD 3-Clause License
 *              See LICENSE file in the project root for full license text.
 */

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <string>

#include <rex/logging.h>

TEST_CASE("another user's folder becomes <user>", "[log_redact]") {
  REQUIRE(rex::RedactUserData("Data folder: D:\\Users\\SomeoneElse\\Saves") == "Data folder: D:\\Users\\<user>\\Saves");
  REQUIRE(rex::RedactUserData("in 'C:/Users/Max Muster/AppData'") == "in 'C:/Users/<user>/AppData'");
  REQUIRE(rex::RedactUserData("path=/home/bob/games") == "path=/home/<user>/games");
}

TEST_CASE("shared folders and ordinary lines stay", "[log_redact]") {
  REQUIRE(rex::RedactUserData("C:\\Users\\Public\\Skylanders") == "C:\\Users\\Public\\Skylanders");
  REQUIRE(rex::RedactUserData("Translated 334 shaders in 8 ms") == "Translated 334 shaders in 8 ms");
  REQUIRE(rex::RedactUserData("") == "");
}

TEST_CASE("the user folder from the environment becomes %USERPROFILE%", "[log_redact]") {
#if defined(_WIN32)
  const char* profile = std::getenv("USERPROFILE");
#else
  const char* profile = std::getenv("HOME");
#endif
  if (!profile || std::string(profile).size() < 4)
    SKIP("no user folder in the environment");
  const std::string line = std::string("Achievements in ") + profile + "\\AppData\\Local";
  REQUIRE(rex::RedactUserData(line) == "Achievements in %USERPROFILE%\\AppData\\Local");
}
