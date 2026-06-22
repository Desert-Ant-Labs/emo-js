// Register the model's binary file extensions as Metro assets so `require()` on
// them returns a bundled asset (resolved to bytes in emo.js).
const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);
config.resolver.assetExts.push("safetensors", "bin");

module.exports = config;
