// Pixel geometry shared by the crop preview and its UI tests. Keep in sync with DonutEditStudio.py.
export const ASPECT_RATIOS = {
    "1:1 Square": [1, 1], "2:3 Portrait": [2, 3], "3:2 Photo": [3, 2],
    "3:4 Portrait": [3, 4], "4:3 Standard": [4, 3], "9:16 Portrait": [9, 16],
    "16:9 Wide": [16, 9], "21:9 Ultrawide": [21, 9],
};
export function targetDimensions(settings, sourceSize, sourceBSize) {
    const grid = Number(settings.multiple || 64), mode = settings.resolution_mode;
    if (mode === "Reference A · crop only" && sourceSize) {
        return sourceSize.map(edge => Math.max(grid, Math.floor(edge / grid) * grid));
    }
    if (mode === "Custom") return [settings.width, settings.height].map(edge => Math.max(grid, Math.round(edge / grid) * grid));
    let aspect = settings.aspect_ratio;
    if (aspect?.startsWith("Auto")) {
        const source = aspect.endsWith("B") ? sourceBSize : sourceSize;
        aspect = source ? Object.keys(ASPECT_RATIOS).reduce((best, key) => {
            const distance = name => Math.abs(Math.log((ASPECT_RATIOS[name][0] / ASPECT_RATIOS[name][1]) / (source[0] / source[1])));
            return distance(key) < distance(best) ? key : best;
        }) : "4:3 Standard";
    }
    const ratio = mode?.startsWith("Reference A") && sourceSize ? sourceSize : ASPECT_RATIOS[aspect];
    const scale = Math.sqrt(settings.megapixels * 1024 * 1024 / (ratio[0] * ratio[1]));
    return ratio.map(edge => Math.max(grid, Math.round(edge * scale / grid) * grid));
}
export function cropBox(sourceWidth, sourceHeight, targetWidth, targetHeight, x = 0.5, y = 0.5, cropOnly = false) {
    let width, height;
    if (cropOnly) { width = Math.min(sourceWidth, targetWidth); height = Math.min(sourceHeight, targetHeight); }
    else if (sourceWidth * targetHeight > sourceHeight * targetWidth) {
        width = Math.max(1, Math.round(sourceHeight * targetWidth / targetHeight)); height = sourceHeight;
    } else { width = sourceWidth; height = Math.max(1, Math.round(sourceWidth * targetHeight / targetWidth)); }
    const left = Math.round((sourceWidth - width) * Math.min(1, Math.max(0, x)));
    const top = Math.round((sourceHeight - height) * Math.min(1, Math.max(0, y)));
    return [left, top, left + width, top + height];
}
export function imageLocation(value) {
    const match = String(value || "").match(/^(.*?)(?:\s+\[(input|output|temp)\])?$/);
    const path = match[1].replaceAll("\\", "/"), slash = path.lastIndexOf("/");
    return { filename: path.slice(slash + 1), subfolder: path.slice(0, Math.max(0, slash)), type: match[2] || "input" };
}
