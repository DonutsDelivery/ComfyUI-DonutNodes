import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Donut LoRA Browser
 * A popup dialog to browse, preview, and select LoRAs with CivitAI info
 */

class DonutLoraBrowser {
    constructor() {
        this.allLoras = [];     // Full list
        this.loras = [];        // Filtered list
        this.currentIndex = 0;
        this.currentInfo = null;
        this.isLoading = false;
        this.dialog = null;
        this.onSelect = null;
        this.imageIndex = 0;    // For cycling through individual images
        this.filterText = "";   // Search filter
    }

    async loadLoraList() {
        try {
            const response = await api.fetchApi("/donut/loras/list");
            if (response.ok) {
                const data = await response.json();
                this.allLoras = data.loras || [];
                this.applyFilter();
                return true;
            }
        } catch (error) {
            console.error("[DonutNodes] Error loading LoRA list:", error);
        }
        return false;
    }

    applyFilter() {
        if (!this.filterText) {
            this.loras = [...this.allLoras];
        } else {
            const filter = this.filterText.toLowerCase();
            this.loras = this.allLoras.filter(l =>
                l.name.toLowerCase().includes(filter) ||
                l.filename.toLowerCase().includes(filter)
            );
        }
        // Clamp current index
        if (this.currentIndex >= this.loras.length) {
            this.currentIndex = Math.max(0, this.loras.length - 1);
        }
    }

    async loadLoraInfo(loraName) {
        this.isLoading = true;
        this.updateUI();

        try {
            const response = await api.fetchApi(`/donut/loras/info?name=${encodeURIComponent(loraName)}`);
            if (response.ok) {
                const data = await response.json();
                this.currentInfo = data;
                this.imageIndex = 0;
            } else {
                this.currentInfo = { name: loraName, error: "Failed to load info" };
            }
        } catch (error) {
            console.error("[DonutNodes] Error loading LoRA info:", error);
            this.currentInfo = { name: loraName, error: error.message };
        }

        this.isLoading = false;
        this.updateUI();
    }

    createDialog() {
        // Create overlay
        const overlay = document.createElement("div");
        overlay.id = "donut-lora-browser-overlay";
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.7);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        // Create dialog
        const dialog = document.createElement("div");
        dialog.id = "donut-lora-browser-dialog";
        dialog.style.cssText = `
            background: #1a1a2e;
            border-radius: 8px;
            padding: 20px;
            min-width: 700px;
            max-width: 900px;
            max-height: 85vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            color: #eee;
            font-family: sans-serif;
        `;

        // Header with title and counter
        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        `;

        const title = document.createElement("h2");
        title.textContent = "LoRA Browser";
        title.style.cssText = "margin: 0; font-size: 18px;";

        const counter = document.createElement("span");
        counter.id = "donut-lora-counter";
        counter.style.cssText = "color: #888; font-size: 14px;";

        header.appendChild(title);
        header.appendChild(counter);
        dialog.appendChild(header);

        // Search box
        const searchContainer = document.createElement("div");
        searchContainer.style.cssText = `
            margin-bottom: 15px;
        `;

        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.id = "donut-lora-search";
        searchInput.placeholder = "Search LoRAs...";
        searchInput.style.cssText = `
            width: 100%;
            padding: 8px 12px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #eee;
            font-size: 13px;
            box-sizing: border-box;
        `;
        searchInput.oninput = (e) => {
            this.filterText = e.target.value;
            this.applyFilter();
            this.currentIndex = 0;
            this.imageIndex = 0;
            if (this.loras.length > 0) {
                this.loadLoraInfo(this.loras[0].name);
            } else {
                this.currentInfo = null;
                this.updateUI();
            }
        };
        // Prevent arrow keys from triggering navigation while typing
        searchInput.onkeydown = (e) => {
            if (e.key === "ArrowLeft" || e.key === "ArrowRight" ||
                e.key === "ArrowUp" || e.key === "ArrowDown") {
                e.stopPropagation();
            }
            if (e.key === "Escape") {
                searchInput.blur();
                e.stopPropagation();
            }
        };

        searchContainer.appendChild(searchInput);
        dialog.appendChild(searchContainer);

        // Navigation
        const nav = document.createElement("div");
        nav.style.cssText = `
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 15px;
            margin-bottom: 15px;
        `;

        const prevBtn = document.createElement("button");
        prevBtn.textContent = "< Prev";
        prevBtn.id = "donut-lora-prev";
        prevBtn.style.cssText = this.getButtonStyle();
        prevBtn.onclick = () => this.navigate(-1);

        const loraName = document.createElement("div");
        loraName.id = "donut-lora-name";
        loraName.style.cssText = `
            flex: 1;
            text-align: center;
            font-size: 14px;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            padding: 0 10px;
        `;

        const nextBtn = document.createElement("button");
        nextBtn.textContent = "Next >";
        nextBtn.id = "donut-lora-next";
        nextBtn.style.cssText = this.getButtonStyle();
        nextBtn.onclick = () => this.navigate(1);

        nav.appendChild(prevBtn);
        nav.appendChild(loraName);
        nav.appendChild(nextBtn);
        dialog.appendChild(nav);

        // Content area (image + info)
        const content = document.createElement("div");
        content.style.cssText = `
            display: flex;
            gap: 20px;
            flex: 1;
            overflow: hidden;
        `;

        // Image area
        const imageArea = document.createElement("div");
        imageArea.style.cssText = `
            width: 300px;
            min-width: 300px;
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 10px;
        `;

        const imageContainer = document.createElement("div");
        imageContainer.id = "donut-lora-image";
        imageContainer.style.cssText = `
            width: 100%;
            height: 300px;
            background: #0d0d1a;
            border-radius: 4px;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        `;

        // Image navigation buttons
        const imageNav = document.createElement("div");
        imageNav.style.cssText = `
            display: flex;
            gap: 10px;
        `;

        const prevImgBtn = document.createElement("button");
        prevImgBtn.textContent = "< Img";
        prevImgBtn.style.cssText = this.getButtonStyle(true);
        prevImgBtn.onclick = () => this.navigateImage(-1);

        const imgCounter = document.createElement("span");
        imgCounter.id = "donut-lora-img-counter";
        imgCounter.style.cssText = "color: #888; font-size: 12px; padding: 5px;";

        const nextImgBtn = document.createElement("button");
        nextImgBtn.textContent = "Img >";
        nextImgBtn.style.cssText = this.getButtonStyle(true);
        nextImgBtn.onclick = () => this.navigateImage(1);

        imageNav.appendChild(prevImgBtn);
        imageNav.appendChild(imgCounter);
        imageNav.appendChild(nextImgBtn);

        imageArea.appendChild(imageContainer);
        imageArea.appendChild(imageNav);
        content.appendChild(imageArea);

        // Info area
        const infoArea = document.createElement("div");
        infoArea.id = "donut-lora-info";
        infoArea.style.cssText = `
            flex: 1;
            overflow-y: auto;
            font-size: 12px;
            line-height: 1.5;
        `;
        content.appendChild(infoArea);

        dialog.appendChild(content);

        // Footer with hints and buttons
        const footer = document.createElement("div");
        footer.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #333;
        `;

        const hints = document.createElement("div");
        hints.style.cssText = "color: #666; font-size: 11px;";
        hints.innerHTML = "<kbd>←</kbd><kbd>→</kbd> LoRAs &nbsp; <kbd>↑</kbd><kbd>↓</kbd> Images &nbsp; <kbd>Enter</kbd> Select &nbsp; <kbd>Esc</kbd> Close";

        const buttons = document.createElement("div");
        buttons.style.cssText = "display: flex; gap: 10px;";

        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = this.getButtonStyle();
        cancelBtn.onclick = () => this.close();

        const civitaiBtn = document.createElement("button");
        civitaiBtn.textContent = "🌐 CivitAI";
        civitaiBtn.style.cssText = this.getButtonStyle() + "background: #3a5a9a;";
        civitaiBtn.onclick = () => {
            this.close();
            if (window.DonutCivitaiBrowser) {
                window.DonutCivitaiBrowser.show();
            }
        };

        const selectBtn = document.createElement("button");
        selectBtn.textContent = "Select LoRA";
        selectBtn.id = "donut-lora-select";
        selectBtn.style.cssText = this.getButtonStyle() + "background: #4a6fa5;";
        selectBtn.onclick = () => this.selectCurrent();

        buttons.appendChild(cancelBtn);
        buttons.appendChild(civitaiBtn);
        buttons.appendChild(selectBtn);
        footer.appendChild(hints);
        footer.appendChild(buttons);
        dialog.appendChild(footer);

        overlay.appendChild(dialog);

        // Close on overlay click
        overlay.onclick = (e) => {
            if (e.target === overlay) this.close();
        };

        // Keyboard navigation
        overlay.tabIndex = 0;
        overlay.onkeydown = (e) => {
            if (e.key === "ArrowLeft") {
                this.navigate(-1);
                e.preventDefault();
            } else if (e.key === "ArrowRight") {
                this.navigate(1);
                e.preventDefault();
            } else if (e.key === "ArrowUp") {
                this.navigateImage(-1);
                e.preventDefault();
            } else if (e.key === "ArrowDown") {
                this.navigateImage(1);
                e.preventDefault();
            } else if (e.key === "Enter") {
                this.selectCurrent();
                e.preventDefault();
            } else if (e.key === "Escape") {
                this.close();
                e.preventDefault();
            }
        };

        this.dialog = overlay;
        return overlay;
    }

    getButtonStyle(small = false) {
        return `
            background: #333;
            color: #eee;
            border: 1px solid #555;
            border-radius: 4px;
            padding: ${small ? "5px 10px" : "8px 15px"};
            cursor: pointer;
            font-size: ${small ? "11px" : "13px"};
            transition: background 0.2s;
        `;
    }

    navigate(direction) {
        if (this.loras.length === 0) return;

        this.currentIndex += direction;
        if (this.currentIndex < 0) this.currentIndex = this.loras.length - 1;
        if (this.currentIndex >= this.loras.length) this.currentIndex = 0;

        this.imageIndex = 0;
        this.loadLoraInfo(this.loras[this.currentIndex].name);
    }

    navigateImage(direction) {
        if (!this.currentInfo?.civitai?.images) return;

        const imageCount = this.currentInfo.civitai.images.length;
        if (imageCount <= 1) return;

        this.imageIndex += direction;
        if (this.imageIndex < 0) this.imageIndex = imageCount - 1;
        if (this.imageIndex >= imageCount) this.imageIndex = 0;

        this.updateImage();
    }

    updateUI() {
        if (!this.dialog) return;

        const counter = this.dialog.querySelector("#donut-lora-counter");
        const nameEl = this.dialog.querySelector("#donut-lora-name");
        const imageEl = this.dialog.querySelector("#donut-lora-image");
        const infoEl = this.dialog.querySelector("#donut-lora-info");

        // Handle empty list
        if (this.loras.length === 0) {
            counter.textContent = "0 / 0";
            nameEl.textContent = "No matches";
            nameEl.title = "";
            if (imageEl) imageEl.innerHTML = '<div style="color: #888;">No LoRAs match your search</div>';
            if (infoEl) infoEl.innerHTML = '<div style="color: #888; text-align: center; padding: 20px;">Try a different search term</div>';
            return;
        }

        // Update counter
        counter.textContent = `${this.currentIndex + 1} / ${this.loras.length}`;

        // Update name
        if (this.loras[this.currentIndex]) {
            nameEl.textContent = this.loras[this.currentIndex].name;
            nameEl.title = this.loras[this.currentIndex].name;
        }

        // Update image
        this.updateImage();

        // Update info
        if (this.isLoading) {
            infoEl.innerHTML = '<div style="text-align: center; padding: 20px;">Loading...</div>';
        } else if (this.currentInfo) {
            infoEl.innerHTML = this.formatInfo(this.currentInfo);
        }
    }

    updateImage() {
        const imageEl = this.dialog?.querySelector("#donut-lora-image");
        const imgCounterEl = this.dialog?.querySelector("#donut-lora-img-counter");
        if (!imageEl) return;

        if (this.isLoading) {
            imageEl.innerHTML = '<div style="color: #888;">Loading...</div>';
            if (imgCounterEl) imgCounterEl.textContent = "";
            return;
        }

        if (this.currentInfo?.hash) {
            // Show individual image based on imageIndex
            const img = document.createElement("img");
            const imageType = this.imageIndex === 0 ? "0" : String(this.imageIndex);
            img.src = api.apiURL(`/donut/loras/preview?hash=${this.currentInfo.hash}&type=${imageType}&t=${Date.now()}`);
            img.style.cssText = "max-width: 100%; max-height: 100%; object-fit: contain;";
            img.onerror = () => {
                // Try collage as fallback
                img.src = api.apiURL(`/donut/loras/preview?hash=${this.currentInfo.hash}&type=collage&t=${Date.now()}`);
                img.onerror = () => {
                    imageEl.innerHTML = '<div style="color: #888;">No preview</div>';
                };
            };
            imageEl.innerHTML = "";
            imageEl.appendChild(img);

            // Update image counter
            if (imgCounterEl && this.currentInfo?.civitai?.images) {
                const total = Math.min(this.currentInfo.civitai.images.length, 4);
                imgCounterEl.textContent = `${this.imageIndex + 1} / ${total}`;
            } else if (imgCounterEl) {
                imgCounterEl.textContent = "";
            }
        } else {
            imageEl.innerHTML = '<div style="color: #888;">No preview</div>';
            if (imgCounterEl) imgCounterEl.textContent = "";
        }
    }

    formatInfo(info) {
        if (info.error && !info.civitai) {
            return `<div style="color: #f88;">${info.error}</div>`;
        }

        const c = info.civitai;
        if (!c) {
            return `
                <div style="color: #888;">
                    <p>No CivitAI info available for this LoRA.</p>
                    <p style="font-size: 11px; margin-top: 10px;">Hash: ${info.hash || "Unknown"}</p>
                </div>
            `;
        }

        let html = "";

        // Model name and version
        html += `<div style="font-size: 14px; font-weight: bold; margin-bottom: 10px;">
            ${c.model_name || "Unknown"}
            <span style="font-weight: normal; color: #888;">(${c.version_name || ""})</span>
        </div>`;

        // Base model
        if (c.base_model) {
            html += `<div style="margin-bottom: 8px;">
                <span style="color: #888;">Base Model:</span>
                <span style="background: #333; padding: 2px 6px; border-radius: 3px;">${c.base_model}</span>
            </div>`;
        }

        // Recommended weight
        if (c.recommended_weight && c.recommended_weight !== 1.0) {
            html += `<div style="margin-bottom: 8px;">
                <span style="color: #888;">Recommended Weight:</span> ${c.recommended_weight}
            </div>`;
        }

        // Trigger words
        if (c.trained_words && c.trained_words.length > 0) {
            html += `<div style="margin-bottom: 8px;">
                <span style="color: #888;">Trigger Words:</span><br>
                <div style="background: #0d0d1a; padding: 8px; border-radius: 4px; margin-top: 4px; font-family: monospace; font-size: 11px;">
                    ${c.trained_words.join(", ")}
                </div>
            </div>`;
        }

        // Stats
        html += `<div style="margin-bottom: 8px; color: #888; font-size: 11px;">
            Downloads: ${c.download_count?.toLocaleString() || 0}
            ${c.rating ? ` | Rating: ${c.rating.toFixed(1)}` : ""}
        </div>`;

        // Description (stripped of HTML)
        if (c.description) {
            const desc = c.description.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
            html += `<div style="margin-bottom: 8px; color: #aaa; font-size: 11px; max-height: 150px; overflow-y: auto; padding-right: 5px;">
                ${desc}
            </div>`;
        }

        // CivitAI link
        if (c.model_url) {
            html += `<div style="margin-top: 10px;">
                <a href="${c.model_url}" target="_blank" style="color: #6a9fd4; text-decoration: none; font-size: 11px;">
                    View on CivitAI
                </a>
            </div>`;
        }

        return html;
    }

    selectCurrent() {
        if (this.loras[this.currentIndex] && this.onSelect) {
            this.onSelect(this.loras[this.currentIndex].name);
        }
        this.close();
    }

    close() {
        if (this.dialog) {
            this.dialog.remove();
            this.dialog = null;
        }
    }

    async show(onSelect, initialLora = null) {
        this.onSelect = onSelect;
        this.filterText = "";  // Reset filter
        this.imageIndex = 0;

        // Load LoRA list
        const loaded = await this.loadLoraList();
        if (!loaded || this.loras.length === 0) {
            alert("No LoRAs found!");
            return;
        }

        // Find initial index if provided
        if (initialLora) {
            const idx = this.loras.findIndex(l => l.name === initialLora);
            if (idx >= 0) this.currentIndex = idx;
        } else {
            this.currentIndex = 0;
        }

        // Create and show dialog
        const dialog = this.createDialog();
        document.body.appendChild(dialog);
        dialog.focus();

        // Load initial info
        this.loadLoraInfo(this.loras[this.currentIndex].name);
    }
}

// Create global instance
const loraBrowser = new DonutLoraBrowser();

// Export for use in other modules
window.DonutLoraBrowser = loraBrowser;


/**
 * Donut LoRA Grid Browser
 * A gallery view to quickly browse and select LoRAs with thumbnails
 */
class DonutLoraGridBrowser {
    constructor() {
        this.allLoras = [];
        this.loras = [];
        this.filterText = "";
        this.folderFilter = "";
        this.folders = [];
        this.dialog = null;
        this.onSelect = null;
        this.selectedIndex = -1;
        this.isFetching = false;
        this.fetchAborted = false;
        this.viewMode = "collage"; // "collage" or "single"
    }

    async loadLoraList() {
        try {
            const response = await api.fetchApi("/donut/loras/list?include_meta=true");
            if (response.ok) {
                const data = await response.json();
                this.allLoras = data.loras || [];

                // Extract unique folders from lora paths
                const folderSet = new Set();
                for (const lora of this.allLoras) {
                    const parts = lora.name.split(/[/\\]/);
                    if (parts.length > 1) {
                        // Add all parent folders
                        let path = "";
                        for (let i = 0; i < parts.length - 1; i++) {
                            path = path ? `${path}/${parts[i]}` : parts[i];
                            folderSet.add(path);
                        }
                    }
                }
                this.folders = Array.from(folderSet).sort();

                this.applyFilter();
                return true;
            }
        } catch (error) {
            console.error("[DonutNodes] Error loading LoRA list:", error);
        }
        return false;
    }

    applyFilter() {
        let filtered = [...this.allLoras];

        // Apply folder filter
        if (this.folderFilter) {
            filtered = filtered.filter(l => {
                const loraFolder = l.name.substring(0, l.name.lastIndexOf('/')) ||
                                   l.name.substring(0, l.name.lastIndexOf('\\')) || "";
                // Match exact folder or subfolders
                return loraFolder === this.folderFilter ||
                       loraFolder.startsWith(this.folderFilter + '/') ||
                       loraFolder.startsWith(this.folderFilter + '\\');
            });
        }

        // Apply text filter
        if (this.filterText) {
            const filter = this.filterText.toLowerCase();
            filtered = filtered.filter(l =>
                l.name.toLowerCase().includes(filter) ||
                l.filename.toLowerCase().includes(filter) ||
                (l.civitai_name && l.civitai_name.toLowerCase().includes(filter))
            );
        }

        this.loras = filtered;
    }

    applyGridStyles(gridContainer) {
        if (this.viewMode === "single") {
            // Masonry-like layout using grid with auto rows for vertical scroll
            gridContainer.style.cssText = `
                flex: 1;
                overflow-y: auto;
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                grid-auto-rows: 10px;
                gap: 0 12px;
                padding: 5px 5px 50px 5px;
                align-items: start;
            `;
        } else {
            // Regular grid for collages (uniform aspect ratio)
            gridContainer.style.cssText = `
                flex: 1;
                overflow-y: auto;
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                grid-auto-rows: min-content;
                gap: 12px;
                padding: 5px;
            `;
        }
    }

    updateGridLayout() {
        const grid = this.dialog?.querySelector("#donut-lora-grid");
        if (grid) {
            this.applyGridStyles(grid);
        }
    }

    createDialog() {
        const overlay = document.createElement("div");
        overlay.id = "donut-lora-grid-overlay";
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        const dialog = document.createElement("div");
        dialog.id = "donut-lora-grid-dialog";
        dialog.style.cssText = `
            background: #1a1a2e;
            border-radius: 8px;
            padding: 20px;
            width: 90vw;
            max-width: 1200px;
            height: 85vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
            color: #eee;
            font-family: sans-serif;
        `;

        // Header
        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
        `;

        const title = document.createElement("h2");
        title.textContent = "LoRA Gallery";
        title.style.cssText = "margin: 0; font-size: 18px;";

        const counter = document.createElement("span");
        counter.id = "donut-grid-counter";
        counter.style.cssText = "color: #888; font-size: 14px;";

        const closeBtn = document.createElement("button");
        closeBtn.innerHTML = "&times;";
        closeBtn.style.cssText = `
            background: none;
            border: none;
            color: #888;
            font-size: 24px;
            cursor: pointer;
            padding: 0 5px;
        `;
        closeBtn.onclick = () => this.close();

        header.appendChild(title);
        header.appendChild(counter);
        header.appendChild(closeBtn);
        dialog.appendChild(header);

        // Search and filter row
        const searchContainer = document.createElement("div");
        searchContainer.style.cssText = "margin-bottom: 15px; display: flex; gap: 10px;";

        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.id = "donut-grid-search";
        searchInput.placeholder = "Search LoRAs by name or CivitAI title...";
        searchInput.style.cssText = `
            flex: 1;
            padding: 10px 15px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #eee;
            font-size: 14px;
            box-sizing: border-box;
        `;
        searchInput.oninput = (e) => {
            this.filterText = e.target.value;
            this.applyFilter();
            this.renderGrid();
        };
        searchInput.onkeydown = (e) => {
            if (e.key === "Escape") {
                if (this.filterText || this.folderFilter) {
                    this.filterText = "";
                    this.folderFilter = "";
                    searchInput.value = "";
                    const folderSelect = this.dialog?.querySelector("#donut-grid-folder");
                    if (folderSelect) folderSelect.value = "";
                    this.applyFilter();
                    this.renderGrid();
                } else {
                    this.close();
                }
                e.preventDefault();
            }
        };

        // Folder filter dropdown
        const folderSelect = document.createElement("select");
        folderSelect.id = "donut-grid-folder";
        folderSelect.style.cssText = `
            padding: 10px 15px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #eee;
            font-size: 14px;
            min-width: 150px;
            cursor: pointer;
        `;
        folderSelect.onchange = (e) => {
            this.folderFilter = e.target.value;
            this.applyFilter();
            this.renderGrid();
        };

        // View mode dropdown
        const viewSelect = document.createElement("select");
        viewSelect.id = "donut-view-mode";
        viewSelect.style.cssText = `
            padding: 10px 15px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #eee;
            font-size: 14px;
            cursor: pointer;
        `;
        const collageOption = document.createElement("option");
        collageOption.value = "collage";
        collageOption.textContent = "Collage";
        const singleOption = document.createElement("option");
        singleOption.value = "single";
        singleOption.textContent = "Single";
        viewSelect.appendChild(collageOption);
        viewSelect.appendChild(singleOption);
        viewSelect.value = this.viewMode;
        viewSelect.onchange = (e) => {
            this.viewMode = e.target.value;
            this.updateGridLayout();
            this.renderGrid();
        };

        // Find Duplicates button
        const duplicatesBtn = document.createElement("button");
        duplicatesBtn.textContent = "Find Duplicates";
        duplicatesBtn.style.cssText = `
            padding: 10px 15px;
            background: #2a2a40;
            border: 1px solid #444;
            border-radius: 4px;
            color: #eee;
            font-size: 14px;
            cursor: pointer;
            margin-left: auto;
        `;
        duplicatesBtn.onmouseover = () => duplicatesBtn.style.background = "#3a3a50";
        duplicatesBtn.onmouseout = () => duplicatesBtn.style.background = "#2a2a40";
        duplicatesBtn.onclick = () => this.showDuplicatesDialog();

        searchContainer.appendChild(searchInput);
        searchContainer.appendChild(folderSelect);
        searchContainer.appendChild(viewSelect);
        searchContainer.appendChild(duplicatesBtn);
        dialog.appendChild(searchContainer);

        // Grid container with vertical scroll
        const gridContainer = document.createElement("div");
        gridContainer.id = "donut-lora-grid";
        this.applyGridStyles(gridContainer);
        dialog.appendChild(gridContainer);

        // Progress bar container (hidden by default)
        const progressContainer = document.createElement("div");
        progressContainer.id = "donut-fetch-progress";
        progressContainer.style.cssText = `
            display: none;
            margin-top: 15px;
            padding: 10px;
            background: #0d0d1a;
            border-radius: 4px;
        `;

        const progressLabel = document.createElement("div");
        progressLabel.id = "donut-fetch-label";
        progressLabel.style.cssText = "font-size: 12px; margin-bottom: 8px; color: #aaa;";
        progressLabel.textContent = "Fetching CivitAI info...";

        const progressBarOuter = document.createElement("div");
        progressBarOuter.style.cssText = `
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            overflow: hidden;
        `;

        const progressBarInner = document.createElement("div");
        progressBarInner.id = "donut-fetch-bar";
        progressBarInner.style.cssText = `
            width: 0%;
            height: 100%;
            background: #4a6fa5;
            transition: width 0.3s;
        `;

        progressBarOuter.appendChild(progressBarInner);
        progressContainer.appendChild(progressLabel);
        progressContainer.appendChild(progressBarOuter);
        dialog.appendChild(progressContainer);

        // Footer
        const footer = document.createElement("div");
        footer.style.cssText = `
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #333;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        `;

        const hints = document.createElement("div");
        hints.id = "donut-grid-hints";
        hints.style.cssText = "color: #666; font-size: 11px; flex: 1;";
        hints.textContent = "Click a LoRA to select it.";

        const buttonsContainer = document.createElement("div");
        buttonsContainer.style.cssText = "display: flex; gap: 10px;";

        // Refresh all checkbox
        const refreshContainer = document.createElement("label");
        refreshContainer.style.cssText = `
            display: flex;
            align-items: center;
            gap: 5px;
            color: #888;
            font-size: 11px;
            cursor: pointer;
        `;

        const refreshCheckbox = document.createElement("input");
        refreshCheckbox.type = "checkbox";
        refreshCheckbox.id = "donut-refresh-all";
        refreshCheckbox.style.cssText = "cursor: pointer;";

        const refreshLabel = document.createElement("span");
        refreshLabel.textContent = "Refresh all";

        refreshContainer.appendChild(refreshCheckbox);
        refreshContainer.appendChild(refreshLabel);

        const fetchAllBtn = document.createElement("button");
        fetchAllBtn.id = "donut-fetch-all-btn";
        fetchAllBtn.textContent = "Fetch Missing";
        fetchAllBtn.style.cssText = `
            background: #2a4a3a;
            color: #eee;
            border: 1px solid #3a6a4a;
            border-radius: 4px;
            padding: 8px 15px;
            cursor: pointer;
            font-size: 13px;
        `;
        fetchAllBtn.onclick = () => this.fetchAllInfo();

        const civitaiBtn = document.createElement("button");
        civitaiBtn.textContent = "🌐 CivitAI Browser";
        civitaiBtn.style.cssText = `
            background: #3a5a9a;
            color: #eee;
            border: 1px solid #4a7aca;
            border-radius: 4px;
            padding: 8px 15px;
            cursor: pointer;
            font-size: 13px;
        `;
        civitaiBtn.onclick = () => {
            this.close();
            if (window.DonutCivitaiBrowser) {
                window.DonutCivitaiBrowser.show();
            }
        };

        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = `
            background: #333;
            color: #eee;
            border: 1px solid #555;
            border-radius: 4px;
            padding: 8px 20px;
            cursor: pointer;
            font-size: 13px;
        `;
        cancelBtn.onclick = () => this.close();

        buttonsContainer.appendChild(refreshContainer);
        buttonsContainer.appendChild(fetchAllBtn);
        buttonsContainer.appendChild(civitaiBtn);
        buttonsContainer.appendChild(cancelBtn);
        footer.appendChild(hints);
        footer.appendChild(buttonsContainer);
        dialog.appendChild(footer);

        overlay.appendChild(dialog);

        // Close on overlay click
        overlay.onclick = (e) => {
            if (e.target === overlay) this.close();
        };

        this.dialog = overlay;
        return overlay;
    }

    renderGrid() {
        const grid = this.dialog?.querySelector("#donut-lora-grid");
        const counter = this.dialog?.querySelector("#donut-grid-counter");
        if (!grid) return;

        grid.innerHTML = "";

        if (counter) {
            counter.textContent = `${this.loras.length} LoRAs`;
        }

        for (let i = 0; i < this.loras.length; i++) {
            const lora = this.loras[i];
            const card = this.createCard(lora, i);
            grid.appendChild(card);
        }
    }

    createCard(lora, index) {
        const card = document.createElement("div");
        card.className = "donut-lora-card";
        card.dataset.loraName = lora.name;  // Store name for easy lookup

        card.style.cssText = `
            background: #0d0d1a;
            border-radius: 6px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.15s, box-shadow 0.15s, border-color 0.15s;
            border: 2px solid transparent;
            position: relative;
            margin-bottom: ${this.viewMode === "single" ? "12px" : "0"};
        `;

        card.onmouseenter = () => {
            card.style.transform = "scale(1.02)";
            card.style.boxShadow = "0 4px 15px rgba(0, 0, 0, 0.4)";
            card.style.borderColor = "#4a6fa5";
        };
        card.onmouseleave = () => {
            card.style.transform = "scale(1)";
            card.style.boxShadow = "none";
            card.style.borderColor = "transparent";
        };

        card.onclick = () => {
            if (this.onSelect) {
                this.onSelect(lora.name);
            }
            this.close();
        };

        // Image container (no fixed height for masonry)
        const imgContainer = document.createElement("div");
        imgContainer.style.cssText = `
            width: 100%;
            background: #151525;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            min-height: 120px;
        `;

        if (lora.hash && lora.has_preview) {
            const img = document.createElement("img");
            // Use collage or single image based on view mode
            const imageType = this.viewMode === "single" ? "0" : "collage";
            img.src = api.apiURL(`/donut/loras/preview?hash=${lora.hash}&type=${imageType}`);
            img.style.cssText = "width: 100%; display: block;";

            // For single view masonry, adjust card height after image loads
            if (this.viewMode === "single") {
                img.onload = () => {
                    // Calculate row span based on actual card height
                    const rowHeight = 10; // matches grid-auto-rows
                    const cardHeight = card.getBoundingClientRect().height;
                    const rowSpan = Math.ceil(cardHeight / rowHeight);
                    card.style.gridRowEnd = `span ${rowSpan}`;
                };
            }

            img.onerror = () => {
                // Fallback: try the other type
                const fallbackType = this.viewMode === "single" ? "collage" : "0";
                img.src = api.apiURL(`/donut/loras/preview?hash=${lora.hash}&type=${fallbackType}`);
                img.onerror = () => {
                    imgContainer.innerHTML = '<div style="color: #555; font-size: 11px; padding: 40px 0;">No preview</div>';
                };
            };
            imgContainer.appendChild(img);
        } else {
            imgContainer.innerHTML = '<div style="color: #555; font-size: 11px; padding: 40px 0;">No preview</div>';
            // For cards without preview in single view
            if (this.viewMode === "single") {
                card.style.gridRowEnd = "span 18"; // ~180px default height
            }
        }

        card.appendChild(imgContainer);

        // Title overlay at the bottom of the image
        const titleOverlay = document.createElement("div");
        titleOverlay.style.cssText = `
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.85));
            padding: 20px 10px 10px 10px;
            color: #fff;
        `;

        // Title (CivitAI name or filename)
        const titleEl = document.createElement("div");
        titleEl.style.cssText = `
            font-size: 12px;
            font-weight: bold;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            text-shadow: 0 1px 3px rgba(0,0,0,0.8);
        `;
        titleEl.textContent = lora.civitai_name || lora.filename.replace(/\.(safetensors|pt|ckpt)$/, "");
        titleEl.title = lora.civitai_name || lora.filename;
        titleOverlay.appendChild(titleEl);

        // Version and base model on same line
        if (lora.civitai_version || lora.base_model) {
            const metaLine = document.createElement("div");
            metaLine.style.cssText = `
                font-size: 10px;
                color: #ccc;
                margin-top: 3px;
                display: flex;
                gap: 8px;
                align-items: center;
            `;

            if (lora.civitai_version) {
                const versionEl = document.createElement("span");
                versionEl.textContent = lora.civitai_version;
                metaLine.appendChild(versionEl);
            }

            if (lora.base_model) {
                const badge = document.createElement("span");
                badge.style.cssText = `
                    font-size: 9px;
                    background: rgba(255,255,255,0.2);
                    padding: 1px 5px;
                    border-radius: 3px;
                `;
                badge.textContent = lora.base_model;
                metaLine.appendChild(badge);
            }

            titleOverlay.appendChild(metaLine);
        }

        card.appendChild(titleOverlay);

        return card;
    }

    updateCard(lora) {
        // Find the card for this lora and update it in place
        const grid = this.dialog?.querySelector("#donut-lora-grid");
        if (!grid) return;

        const existingCard = grid.querySelector(`[data-lora-name="${CSS.escape(lora.name)}"]`);
        if (!existingCard) return;

        // Find index in current filtered list
        const index = this.loras.findIndex(l => l.name === lora.name);
        if (index === -1) return;

        // Create new card and replace old one
        const newCard = this.createCard(lora, index);
        existingCard.replaceWith(newCard);
    }

    close() {
        // Abort any ongoing fetch
        this.fetchAborted = true;
        this.isFetching = false;

        if (this.dialog) {
            this.dialog.remove();
            this.dialog = null;
        }
    }

    async fetchAllInfo() {
        if (this.isFetching) {
            // Already fetching - abort
            this.fetchAborted = true;
            return;
        }

        // Check if we should refresh all or just missing
        const refreshAll = this.dialog?.querySelector("#donut-refresh-all")?.checked || false;

        // Find LoRAs to fetch
        const lorasToFetch = refreshAll
            ? [...this.allLoras]  // All loras
            : this.allLoras.filter(l => !l.has_preview);  // Only missing

        if (lorasToFetch.length === 0) {
            const hints = this.dialog?.querySelector("#donut-grid-hints");
            if (hints) hints.textContent = "All LoRAs already have CivitAI info cached!";
            return;
        }

        this.isFetching = true;
        this.fetchAborted = false;

        // Show progress bar
        const progressContainer = this.dialog?.querySelector("#donut-fetch-progress");
        const progressBar = this.dialog?.querySelector("#donut-fetch-bar");
        const progressLabel = this.dialog?.querySelector("#donut-fetch-label");
        const fetchBtn = this.dialog?.querySelector("#donut-fetch-all-btn");
        const hints = this.dialog?.querySelector("#donut-grid-hints");

        if (progressContainer) progressContainer.style.display = "block";
        if (fetchBtn) {
            fetchBtn.textContent = "Stop";
            fetchBtn.style.background = "#4a2a2a";
            fetchBtn.style.borderColor = "#6a3a3a";
        }

        let completed = 0;
        let found = 0;
        let notFound = 0;
        let errors = 0;
        const total = lorasToFetch.length;

        // Check if API key is configured for faster rate limits
        let hasApiKey = false;
        try {
            const configResp = await api.fetchApi("/donut/config");
            if (configResp.ok) {
                const config = await configResp.json();
                hasApiKey = !!(config.civitai?.api_key);
            }
        } catch (e) {
            // Ignore, use default rate limit
        }

        // With API key: 3 parallel requests, 200ms delay between batches
        // Without API key: 2 parallel requests, 500ms delay between batches
        const PARALLEL = hasApiKey ? 3 : 2;
        const DELAY_MS = hasApiKey ? 200 : 500;
        const rateInfo = hasApiKey ? `(fast: ${PARALLEL}x parallel)` : `(${PARALLEL}x parallel)`;

        // Process in batches
        for (let i = 0; i < lorasToFetch.length; i += PARALLEL) {
            if (this.fetchAborted) {
                break;
            }

            const batch = lorasToFetch.slice(i, i + PARALLEL);

            // Update progress
            if (progressLabel) {
                progressLabel.textContent = `Fetching ${rateInfo}: ${batch[0].filename} (${Math.min(i + PARALLEL, total)}/${total})`;
            }
            if (progressBar) {
                progressBar.style.width = `${(Math.min(i + PARALLEL, total) / total) * 100}%`;
            }

            // Fetch batch in parallel
            const results = await Promise.all(batch.map(async (lora) => {
                try {
                    const response = await api.fetchApi(`/donut/loras/info?name=${encodeURIComponent(lora.name)}`);
                    if (response.ok) {
                        const data = await response.json();
                        if (data.civitai) {
                            // Update the lora in our list
                            lora.civitai_name = data.civitai.model_name;
                            lora.civitai_version = data.civitai.version_name;
                            lora.base_model = data.civitai.base_model;
                            lora.has_preview = true;
                            lora.hash = data.hash;

                            // Update the card immediately in the grid
                            this.updateCard(lora);
                            return "found";
                        } else {
                            return "notFound";
                        }
                    } else {
                        return "error";
                    }
                } catch (e) {
                    console.error(`[DonutNodes] Error fetching info for ${lora.name}:`, e);
                    return "error";
                }
            }));

            // Count results
            for (const result of results) {
                if (result === "found") found++;
                else if (result === "notFound") notFound++;
                else errors++;
            }

            completed += batch.length;

            // Rate limiting delay between batches (skip on last batch or if aborted)
            if (!this.fetchAborted && i + PARALLEL < lorasToFetch.length) {
                await new Promise(resolve => setTimeout(resolve, DELAY_MS));
            }
        }

        // Done fetching
        this.isFetching = false;

        // Update UI
        if (progressContainer) progressContainer.style.display = "none";
        if (fetchBtn) {
            fetchBtn.textContent = "Fetch All Info";
            fetchBtn.style.background = "#2a4a3a";
            fetchBtn.style.borderColor = "#3a6a4a";
        }

        // Show results
        if (hints) {
            if (this.fetchAborted) {
                hints.textContent = `Stopped. Found: ${found}, Not on CivitAI: ${notFound}, Errors: ${errors}`;
            } else {
                hints.textContent = `Done! Found: ${found}, Not on CivitAI: ${notFound}, Errors: ${errors}`;
            }
        }

        // Re-apply filter and re-render grid to show updated info
        this.applyFilter();
        this.renderGrid();

        this.fetchAborted = false;
    }

    populateFolderDropdown() {
        const folderSelect = this.dialog?.querySelector("#donut-grid-folder");
        if (!folderSelect) return;

        folderSelect.innerHTML = "";

        // Add "All folders" option
        const allOption = document.createElement("option");
        allOption.value = "";
        allOption.textContent = "All folders";
        folderSelect.appendChild(allOption);

        // Add folder options
        for (const folder of this.folders) {
            const option = document.createElement("option");
            option.value = folder;
            // Indent subfolders for visual hierarchy
            const depth = (folder.match(/[/\\]/g) || []).length;
            option.textContent = "  ".repeat(depth) + folder.split(/[/\\]/).pop();
            option.title = folder;
            folderSelect.appendChild(option);
        }
    }

    async show(onSelect, initialFilter = "") {
        this.onSelect = onSelect;
        this.filterText = initialFilter;
        this.folderFilter = "";

        // Show loading state
        const loadingOverlay = document.createElement("div");
        loadingOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
            color: #eee;
            font-size: 16px;
        `;
        loadingOverlay.textContent = "Loading LoRAs...";
        document.body.appendChild(loadingOverlay);

        const loaded = await this.loadLoraList();
        loadingOverlay.remove();

        if (!loaded || this.allLoras.length === 0) {
            alert("No LoRAs found!");
            return;
        }

        const dialog = this.createDialog();
        document.body.appendChild(dialog);

        // Populate folder dropdown
        this.populateFolderDropdown();

        // Set initial filter if provided
        if (initialFilter) {
            const searchInput = dialog.querySelector("#donut-grid-search");
            if (searchInput) searchInput.value = initialFilter;
        }

        this.renderGrid();

        // Focus search
        const searchInput = dialog.querySelector("#donut-grid-search");
        if (searchInput) searchInput.focus();
    }

    async showDuplicatesDialog() {
        // Create overlay
        const overlay = document.createElement("div");
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.8);
            z-index: 10002;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        const dialog = document.createElement("div");
        dialog.style.cssText = `
            background: #1a1a2e;
            border-radius: 8px;
            padding: 20px;
            max-width: 700px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
            color: #eee;
        `;

        dialog.innerHTML = `
            <h2 style="margin: 0 0 15px 0; font-size: 18px;">Finding Duplicate LoRAs...</h2>
            <p style="color: #aaa; margin: 0;">Scanning for files with identical hashes...</p>
        `;

        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        // Close on overlay click
        overlay.onclick = (e) => {
            if (e.target === overlay) overlay.remove();
        };

        // Fetch duplicates
        try {
            const response = await api.fetchApi("/donut/loras/duplicates");
            if (!response.ok) {
                throw new Error("Failed to fetch duplicates");
            }

            const data = await response.json();

            if (data.totalGroups === 0) {
                dialog.innerHTML = `
                    <h2 style="margin: 0 0 15px 0; font-size: 18px;">No Duplicates Found</h2>
                    <p style="color: #aaa;">All your LoRAs have unique hashes.</p>
                    <button id="close-dup-btn" style="margin-top: 15px; padding: 10px 20px; background: #444; border: none; border-radius: 4px; color: #eee; cursor: pointer;">Close</button>
                `;
                dialog.querySelector("#close-dup-btn").onclick = () => overlay.remove();
                return;
            }

            // Format file size
            const formatSize = (bytes) => {
                if (bytes < 1024) return bytes + " B";
                if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
                if (bytes < 1024 * 1024 * 1024) return (bytes / (1024 * 1024)).toFixed(1) + " MB";
                return (bytes / (1024 * 1024 * 1024)).toFixed(2) + " GB";
            };

            let html = `
                <h2 style="margin: 0 0 10px 0; font-size: 18px;">Duplicate LoRAs Found</h2>
                <p style="color: #aaa; margin: 0 0 15px 0;">
                    Found <b>${data.totalGroups}</b> groups with <b>${data.totalDuplicateFiles}</b> duplicate files
                    (${formatSize(data.totalWastedBytes)} wasted)
                </p>
                <div style="max-height: 50vh; overflow-y: auto;">
            `;

            for (const group of data.duplicates) {
                html += `
                    <div style="background: #252540; border-radius: 4px; padding: 12px; margin-bottom: 10px;">
                        <div style="font-size: 11px; color: #666; margin-bottom: 8px;">Hash: ${group.sha256.substring(0, 16)}...</div>
                `;

                for (let i = 0; i < group.files.length; i++) {
                    const file = group.files[i];
                    const isFirst = i === 0;
                    html += `
                        <div style="display: flex; align-items: center; gap: 10px; padding: 6px 0; ${!isFirst ? 'border-top: 1px solid #333;' : ''}">
                            <span style="flex: 1; font-size: 13px; ${isFirst ? 'color: #8f8;' : 'color: #f88;'}">${file.filename}</span>
                            <span style="color: #888; font-size: 12px;">${formatSize(file.size)}</span>
                            ${!isFirst ? `<button class="delete-dup-btn" data-path="${file.full_path}" style="padding: 4px 10px; background: #a33; border: none; border-radius: 3px; color: #fff; cursor: pointer; font-size: 12px;">Delete</button>` : '<span style="color: #8f8; font-size: 11px;">KEEP</span>'}
                        </div>
                    `;
                }

                html += `</div>`;
            }

            html += `
                </div>
                <div style="display: flex; gap: 10px; margin-top: 15px;">
                    <button id="close-dup-btn" style="padding: 10px 20px; background: #444; border: none; border-radius: 4px; color: #eee; cursor: pointer;">Close</button>
                </div>
            `;

            dialog.innerHTML = html;

            // Add close handler
            dialog.querySelector("#close-dup-btn").onclick = () => overlay.remove();

            // Add delete handlers
            dialog.querySelectorAll(".delete-dup-btn").forEach(btn => {
                btn.onclick = async (e) => {
                    const path = btn.dataset.path;
                    if (!confirm(`Delete this duplicate?\n\n${path}`)) return;

                    btn.disabled = true;
                    btn.textContent = "...";

                    try {
                        const delResponse = await api.fetchApi(`/donut/loras/by-path?path=${encodeURIComponent(path)}`, {
                            method: "DELETE"
                        });

                        if (delResponse.ok) {
                            btn.parentElement.style.opacity = "0.4";
                            btn.textContent = "Deleted";
                            btn.style.background = "#333";
                        } else {
                            btn.textContent = "Error";
                            btn.disabled = false;
                        }
                    } catch (err) {
                        console.error("Delete error:", err);
                        btn.textContent = "Error";
                        btn.disabled = false;
                    }
                };
            });

        } catch (error) {
            console.error("Error fetching duplicates:", error);
            dialog.innerHTML = `
                <h2 style="margin: 0 0 15px 0; font-size: 18px;">Error</h2>
                <p style="color: #f88;">Failed to scan for duplicates: ${error.message}</p>
                <button id="close-dup-btn" style="margin-top: 15px; padding: 10px 20px; background: #444; border: none; border-radius: 4px; color: #eee; cursor: pointer;">Close</button>
            `;
            dialog.querySelector("#close-dup-btn").onclick = () => overlay.remove();
        }
    }
}

// Create global grid browser instance
const loraGridBrowser = new DonutLoraGridBrowser();
window.DonutLoraGridBrowser = loraGridBrowser;

// Register extension to add context menu option and gallery buttons
app.registerExtension({
    name: "donut.LoraBrowser",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "DonutLoRAStack") {
            // Add right-click menu option
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (origGetExtraMenuOptions) {
                    origGetExtraMenuOptions.apply(this, arguments);
                }

                // Add separator
                options.push(null);

                // Add gallery option for each slot
                for (let i = 1; i <= 3; i++) {
                    const slotNum = i;
                    options.push({
                        content: `LoRA Gallery (Slot ${slotNum})`,
                        callback: () => {
                            const loraWidget = this.widgets?.find(w => w.name === `lora_name_${slotNum}`);

                            loraGridBrowser.show((selectedLora) => {
                                if (loraWidget) {
                                    loraWidget.value = selectedLora;
                                    const switchWidget = this.widgets?.find(w => w.name === `switch_${slotNum}`);
                                    if (switchWidget) {
                                        switchWidget.value = "On";
                                    }
                                    this.setDirtyCanvas(true);
                                }
                            });
                        }
                    });
                }

                // Add separator
                options.push(null);

                // Add single-LoRA browser option for each slot (for detailed view)
                for (let i = 1; i <= 3; i++) {
                    const slotNum = i;
                    options.push({
                        content: `Browse LoRAs (Slot ${slotNum})`,
                        callback: () => {
                            const loraWidget = this.widgets?.find(w => w.name === `lora_name_${slotNum}`);
                            const currentLora = loraWidget?.value;

                            loraBrowser.show((selectedLora) => {
                                if (loraWidget) {
                                    loraWidget.value = selectedLora;
                                    // Also enable the slot
                                    const switchWidget = this.widgets?.find(w => w.name === `switch_${slotNum}`);
                                    if (switchWidget) {
                                        switchWidget.value = "On";
                                    }
                                    this.setDirtyCanvas(true);
                                }
                            }, currentLora !== "None" ? currentLora : null);
                        }
                    });
                }
            };
        }
    },

    nodeCreated(node) {
        if (node.comfyClass === "DonutLoRAStack") {
            // Override LoRA dropdown clicks to open gallery instead
            for (let i = 1; i <= 3; i++) {
                const slotNum = i;
                const loraWidget = node.widgets?.find(w => w.name === `lora_name_${slotNum}`);

                if (loraWidget) {
                    // Store original callback
                    const origCallback = loraWidget.callback;

                    // Override the widget's mouse handler to intercept clicks
                    const origMouse = loraWidget.mouse;
                    loraWidget.mouse = function(event, pos, node) {
                        if (event.type === "pointerdown") {
                            // Open gallery instead of dropdown
                            loraGridBrowser.show((selectedLora) => {
                                this.value = selectedLora;
                                const switchWidget = node.widgets?.find(w => w.name === `switch_${slotNum}`);
                                if (switchWidget) {
                                    switchWidget.value = "On";
                                }
                                if (origCallback) {
                                    origCallback.call(this, selectedLora);
                                }
                                node.setDirtyCanvas(true);
                            });
                            return true; // Consume the event, don't show dropdown
                        }
                        if (origMouse) {
                            return origMouse.call(this, event, pos, node);
                        }
                        return false;
                    };
                }
            }
        }
    }
});
