export function clipboardImage(data) {
    const direct = [...(data?.files || [])].find(file => file?.type?.startsWith("image/"));
    if (direct) return direct;
    for (const item of [...(data?.items || [])]) {
        if (!item?.type?.startsWith("image/")) continue;
        const file = item.getAsFile?.();
        if (file) return file;
    }
    return null;
}

export async function readClipboardImage(clipboard = navigator.clipboard) {
    if (typeof clipboard?.read !== "function") {
        throw new DOMException("Clipboard image reads are unavailable", "NotSupportedError");
    }
    const entries = await clipboard.read();
    for (const item of entries) {
        const type = [...(item.types || [])].find(type => type.startsWith("image/"));
        if (type) return item.getType(type);
    }
    return null;
}
