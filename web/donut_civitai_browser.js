import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * Donut CivitAI Browser
 * Browse, search, and download models from CivitAI directly in ComfyUI
 */

class DonutCivitaiBrowser {
    constructor() {
        this.dialog = null;
        this.currentView = "grid"; // "grid" or "detail"
        this.searchResults = [];
        this.currentModel = null;
        this.currentPage = 1;
        this.hasNextPage = false;
        this.nextCursor = null;
        this.isLoading = false;
        this.searchTimeout = null;
        this.filterTimeout = null;  // Debounce for filter changes
        this.pendingSearch = false;  // Track if a search is queued
        this.activeDownloads = {};

        // Target node for "Download & Load" feature
        this.targetNode = null;  // The DonutLoRAStack node that opened the browser

        // Fetch missing info state
        this.isFetchingMissing = false;
        this.fetchMissingAborted = false;

        // Pagination state
        this.cursors = [];  // Stack of cursors for previous pages
        this.nextCursor = null;  // Cursor for next page

        // Default filter state
        this.defaultFilters = {
            query: "",
            types: [],  // Empty = all types
            sort: "Highest Rated",
            period: "AllTime",
            nsfw: false,
            baseModels: [],
            tag: "",
            username: "",
            downloadedOnly: false,  // Show only locally downloaded models
            updatesOnly: false      // Show only models with updates available
        };

        // Cache for models with updates available (populated by checkForUpdates)
        this.modelsWithUpdates = new Map();  // model_id -> update info
        this.isCheckingUpdates = false;

        // Endless scroll setting
        this.endlessScroll = this.loadEndlessScrollSetting();

        // Model types - display names that match CivitAI's UI
        // The API values are the same as display names for types
        // Note: "LoCon" is the correct API value (not "LyCORIS")
        this.modelTypes = [
            "Checkpoint",
            "Embedding",
            "Hypernetwork",
            "AestheticGradient",
            "LORA",
            "LoCon",
            "DoRA",
            "Controlnet",
            "Upscaler",
            "Motion",
            "VAE",
            "Poses",
            "Wildcards",
            "Workflows",
            "Detection",
            "Other"
        ];
        this.sortOptions = ["Highest Rated", "Most Downloaded", "Newest"];
        this.periodOptions = ["AllTime", "Year", "Month", "Week", "Day"];

        // Display name to API value mapping for base models
        // Key = display name (shown in UI matching CivitAI's website)
        // Value = API value (sent to CivitAI API)
        // Only list names that differ between UI and API
        this.baseModelDisplayToApi = {
            // Spaced vs camelCase/joined differences
            "Z Image Turbo": "ZImageTurbo",
            "Aura Flow": "AuraFlow",
            "Cog Video X": "CogVideoX",
            "Hi Dream": "HiDream",
            "LTX Video": "LTXV",
            // Flux naming (CivitAI UI shows "Dev"/"Schnell", API uses "D"/"S")
            "Flux.1 Dev": "Flux.1 D",
            "Flux.1 Schnell": "Flux.1 S",
            // PixArt naming
            "PixArt A": "PixArt a",
            "PixArt Σ": "PixArt E",
            // Video models
            "Stable Video Diffusion": "SVD",
        };

        // Reverse mapping: API value to display name
        this.baseModelApiToDisplay = {};
        for (const [display, api] of Object.entries(this.baseModelDisplayToApi)) {
            this.baseModelApiToDisplay[api] = display;
        }

        // Fallback base models - display names matching CivitAI's UI
        // The mapping above converts these to API values when searching
        // These will be auto-discovered from API and mapped to display names
        this.fallbackBaseModels = [
            "Aura Flow",
            "Chroma",
            "Cog Video X",
            "Flux.1 Dev",
            "Flux.1 Schnell",
            "Hi Dream",
            "Hunyuan 1",
            "Hunyuan Video",
            "Illustrious",
            "Kolors",
            "LTX Video",
            "Lumina",
            "Mochi",
            "NoobAI",
            "PixArt A",
            "PixArt Σ",
            "Pony",
            "SD 1.4",
            "SD 1.5",
            "SD 1.5 Hyper",
            "SD 1.5 LCM",
            "SD 2.0",
            "SD 2.1",
            "SD 3",
            "SD 3.5",
            "SD 3.5 Large",
            "SD 3.5 Medium",
            "SDXL 0.9",
            "SDXL 1.0",
            "SDXL 1.0 LCM",
            "SDXL Hyper",
            "SDXL Lightning",
            "SDXL Turbo",
            "Stable Cascade",
            "Stable Video Diffusion",
            "Wan Video",
            "Z Image Turbo",
            "Other"
        ];

        // Will be populated from API or cache
        this.baseModelOptions = this.loadCachedBaseModels() || [...this.fallbackBaseModels];

        // Load saved filters or use defaults (must be after baseModelOptions/fallbackBaseModels are defined)
        this.filters = this.loadFilters();

        // Fetch fresh base models in background
        this.fetchBaseModelsFromAPI();

        // Callback for when a model is selected (for node integration)
        this.onSelect = null;

        // Track if API key is configured
        this.hasApiKey = false;
        this.checkApiKey();

        // Set of local SHA256 hashes for checking downloaded status
        this.localHashes = new Set();
        this.localHashesLoaded = false;

        // Saved scroll position for returning from detail view
        this.savedScrollPosition = 0;
    }

    async checkApiKey() {
        try {
            const response = await api.fetchApi("/donut/config");
            if (response.ok) {
                const config = await response.json();
                this.hasApiKey = !!(config.civitai?.api_key);
            }
        } catch (e) {
            console.log("[CivitAI Browser] Could not check API key status");
        }
    }

    async loadLocalHashes(forceReload = false) {
        // Load SHA256 hashes from all local LoRAs for downloaded status checking
        if (this.localHashesLoaded && !forceReload) return;

        try {
            const response = await api.fetchApi("/donut/loras/hashes");
            if (response.ok) {
                const data = await response.json();
                this.localHashes = new Set(data.hashes || []);
                this.localHashesLoaded = true;
                console.log(`[CivitAI Browser] Loaded ${this.localHashes.size} local hashes`);
            } else {
                console.log("[CivitAI Browser] Failed to load hashes, status:", response.status);
            }
        } catch (e) {
            console.log("[CivitAI Browser] Could not load local hashes:", e);
        }
    }

    isModelDownloaded(modelVersion) {
        // Check if any file in this model version matches a local hash
        if (!modelVersion?.files || this.localHashes.size === 0) return false;

        for (const file of modelVersion.files) {
            const sha256 = file.hashes?.SHA256;
            if (sha256 && this.localHashes.has(sha256.toUpperCase())) {
                return true;
            }
        }
        return false;
    }

    // Mark a newly downloaded hash as local (so UI updates without reload)
    addLocalHash(sha256) {
        if (sha256) {
            this.localHashes.add(sha256.toUpperCase());
        }
    }

    // Remove a hash from local set (when file is deleted)
    removeLocalHash(sha256) {
        if (sha256) {
            this.localHashes.delete(sha256.toUpperCase());
        }
    }

    async deleteLora(sha256, modelName) {
        // Delete a downloaded LoRA by its hash
        if (!sha256) return false;

        // Confirm deletion
        if (!confirm(`Are you sure you want to delete "${modelName}"?\n\nThis will permanently remove the file from disk.`)) {
            return false;
        }

        try {
            const response = await api.fetchApi(`/donut/loras/by-hash?sha256=${encodeURIComponent(sha256)}`, {
                method: "DELETE"
            });

            if (!response.ok) {
                const error = await response.json();
                this.showNotification("Delete Failed", error.error || "Unknown error", 5000);
                return false;
            }

            const data = await response.json();
            if (data.deleted) {
                // Remove from local hashes
                this.removeLocalHash(sha256);
                // Refresh folder cache
                await this.refreshFolderCache("LORA");
                this.showNotification("Deleted", data.filename, 3000);
                // Refresh detail view to update buttons
                if (this.currentModel?.id) {
                    this.loadModelDetails(this.currentModel.id);
                }
                return true;
            } else {
                this.showNotification("Delete Failed", data.error || "File not found", 5000);
                return false;
            }
        } catch (error) {
            console.error("[CivitAI Browser] Error deleting LoRA:", error);
            this.showNotification("Delete Failed", error.message, 5000);
            return false;
        }
    }

    loadCachedBaseModels() {
        // Load cached base models from localStorage
        try {
            const cached = localStorage.getItem("donut-civitai-base-models");
            if (cached) {
                const data = JSON.parse(cached);
                // Check if cache is still valid (24 hour TTL)
                const cacheAge = Date.now() - (data.timestamp || 0);
                const ONE_DAY = 24 * 60 * 60 * 1000;

                // Also check cache version - invalidate old caches that might have bad data
                const CACHE_VERSION = 4;  // Increment this to invalidate old caches
                if (data.version !== CACHE_VERSION) {
                    console.log("[CivitAI Browser] Cache version mismatch, clearing old cache");
                    localStorage.removeItem("donut-civitai-base-models");
                    return null;
                }

                if (cacheAge < ONE_DAY && data.models && data.models.length > 0) {
                    console.log(`[CivitAI Browser] Loaded ${data.models.length} base models from cache`);
                    return data.models;
                }
            }
        } catch (e) {
            console.error("[CivitAI Browser] Error loading cached base models:", e);
        }
        return null;
    }

    saveCachedBaseModels(models) {
        try {
            localStorage.setItem("donut-civitai-base-models", JSON.stringify({
                models: models,
                timestamp: Date.now(),
                version: 4  // Must match CACHE_VERSION in loadCachedBaseModels
            }));
        } catch (e) {
            console.error("[CivitAI Browser] Error saving base models cache:", e);
        }
    }

    async fetchBaseModelsFromAPI() {
        // Skip if we already have a valid cache
        const cached = localStorage.getItem("donut-civitai-base-models");
        if (cached) {
            try {
                const data = JSON.parse(cached);
                const cacheAge = Date.now() - (data.timestamp || 0);
                const ONE_DAY = 24 * 60 * 60 * 1000;
                if (cacheAge < ONE_DAY) {
                    // Cache is still valid, no need to fetch
                    return;
                }
            } catch (e) {}
        }

        // Fetch base models by querying different model types and extracting unique baseModels
        try {
            console.log("[CivitAI Browser] Fetching base models from CivitAI...");

            // Query for popular models of different types to get a good spread of base models
            const queries = [
                { types: ["LORA"], sort: "Most Downloaded", limit: 100 },
                { types: ["Checkpoint"], sort: "Most Downloaded", limit: 50 },
            ];

            const baseModelsSet = new Set(this.fallbackBaseModels);

            for (const query of queries) {
                try {
                    const params = new URLSearchParams({
                        sort: query.sort,
                        limit: query.limit.toString(),
                        nsfw: "true"  // Include all to get more variety
                    });

                    for (const type of query.types) {
                        params.append("types", type);
                    }

                    const response = await api.fetchApi(`/donut/civitai/search?${params}`);
                    if (response.ok) {
                        const data = await response.json();
                        const items = data.items || [];

                        for (const item of items) {
                            // Get base model from each version
                            const versions = item.modelVersions || [];
                            for (const version of versions) {
                                if (version.baseModel && version.baseModel.trim()) {
                                    // Convert API value to display name before storing
                                    const apiValue = version.baseModel.trim();
                                    const displayName = this.apiToDisplayBaseModel(apiValue);
                                    baseModelsSet.add(displayName);
                                }
                            }
                        }
                    }
                } catch (e) {
                    console.warn("[CivitAI Browser] Error fetching models for base model discovery:", e);
                }
            }

            // Convert to alphabetically sorted array with "Other" at the end
            // These are already display names from the conversion above
            const baseModels = Array.from(baseModelsSet)
                .filter(m => m !== "Other")
                .sort((a, b) => a.localeCompare(b));

            // Always add "Other" at the end
            baseModels.push("Other");

            if (baseModels.length > this.fallbackBaseModels.length) {
                console.log(`[CivitAI Browser] Discovered ${baseModels.length} base models (was ${this.fallbackBaseModels.length})`);
                this.baseModelOptions = baseModels;
                this.saveCachedBaseModels(baseModels);

                // If sidebar is open, update it
                if (this.dialog) {
                    const oldSidebar = document.getElementById("donut-civitai-sidebar");
                    if (oldSidebar) {
                        const newSidebar = this.createSidebar();
                        oldSidebar.replaceWith(newSidebar);
                    }
                }
            }
        } catch (e) {
            console.error("[CivitAI Browser] Error fetching base models from API:", e);
        }
    }

    loadFilters() {
        // Load saved filters from localStorage
        try {
            const saved = localStorage.getItem("donut-civitai-filters");
            if (saved) {
                const parsed = JSON.parse(saved);
                // Merge with defaults to ensure all fields exist
                const filters = { ...this.defaultFilters, ...parsed };

                // Validate baseModels - remove any that aren't in our known list
                // This prevents issues with invalid cached base models
                if (filters.baseModels && filters.baseModels.length > 0) {
                    const validBaseModels = filters.baseModels.filter(bm =>
                        this.baseModelOptions.includes(bm) || this.fallbackBaseModels.includes(bm)
                    );
                    if (validBaseModels.length !== filters.baseModels.length) {
                        console.log("[CivitAI Browser] Removed invalid base models from saved filters");
                        filters.baseModels = validBaseModels;
                    }
                }

                return filters;
            }
        } catch (e) {
            console.error("[CivitAI Browser] Error loading filters:", e);
        }
        return { ...this.defaultFilters };
    }

    saveFilters() {
        // Save current filters to localStorage
        try {
            localStorage.setItem("donut-civitai-filters", JSON.stringify(this.filters));
        } catch (e) {
            console.error("[CivitAI Browser] Error saving filters:", e);
        }
    }

    clearFilters() {
        // Reset all filters to defaults
        this.filters = { ...this.defaultFilters };
        this.resetPagination();
        this.saveFilters();
        // Re-render the sidebar and search
        if (this.dialog) {
            const oldSidebar = document.getElementById("donut-civitai-sidebar");
            if (oldSidebar) {
                const newSidebar = this.createSidebar();
                oldSidebar.replaceWith(newSidebar);
            }
            // Also update the search input
            const searchInput = document.getElementById("donut-civitai-search");
            if (searchInput) {
                searchInput.value = "";
            }
            this.search();
        }
    }

    hasActiveFilters() {
        // Check if any filters are different from defaults
        return (
            this.filters.query !== "" ||
            this.filters.types.length > 0 ||
            this.filters.sort !== "Highest Rated" ||
            this.filters.period !== "AllTime" ||
            this.filters.nsfw !== false ||
            this.filters.baseModels.length > 0 ||
            this.filters.tag !== "" ||
            this.filters.username !== ""
        );
    }

    resetPagination() {
        // Reset pagination state when filters change
        this.currentPage = 1;
        this.cursors = [];
        this.nextCursor = null;
    }

    // Convert display name to API value for base models
    displayToApiBaseModel(displayName) {
        // If there's a mapping, use it; otherwise assume they're the same
        return this.baseModelDisplayToApi[displayName] || displayName;
    }

    // Convert API value to display name for base models
    apiToDisplayBaseModel(apiValue) {
        // If there's a mapping, use it; otherwise assume they're the same
        return this.baseModelApiToDisplay[apiValue] || apiValue;
    }

    debouncedSearch() {
        // Debounce filter changes to avoid rapid API calls
        if (this.filterTimeout) {
            clearTimeout(this.filterTimeout);
        }
        this.filterTimeout = setTimeout(() => {
            this.search();
        }, 150);
    }

    loadEndlessScrollSetting() {
        try {
            const saved = localStorage.getItem("donut-civitai-endless-scroll");
            return saved === "true";
        } catch (e) {
            return false;
        }
    }

    saveEndlessScrollSetting() {
        try {
            localStorage.setItem("donut-civitai-endless-scroll", this.endlessScroll.toString());
        } catch (e) {
            console.error("[CivitAI Browser] Error saving endless scroll setting:", e);
        }
    }

    toggleEndlessScroll() {
        this.endlessScroll = !this.endlessScroll;
        this.saveEndlessScrollSetting();
        // Re-render grid to show/hide pagination
        this.renderGrid();
    }

    async loadMore() {
        // Load more results for endless scroll
        if (this.isLoading || !this.hasNextPage || !this.nextCursor) return;

        this.isLoading = true;
        this.updateLoadingState();

        try {
            const params = new URLSearchParams({
                query: this.filters.query,
                sort: this.filters.sort,
                period: this.filters.period,
                nsfw: this.filters.nsfw.toString(),
                limit: "20",
                cursor: this.nextCursor
            });

            // CivitAI expects repeated params: types=LORA&types=LoCon
            if (this.filters.types.length > 0) {
                for (const type of this.filters.types) {
                    params.append("types", type);
                }
            }

            // CivitAI expects repeated params for baseModels too
            // Convert display names to API values before sending
            if (this.filters.baseModels.length > 0) {
                for (const baseModel of this.filters.baseModels) {
                    const apiValue = this.displayToApiBaseModel(baseModel);
                    params.append("baseModels", apiValue);
                }
            }

            if (this.filters.tag) {
                params.set("tag", this.filters.tag);
            }

            if (this.filters.username) {
                params.set("username", this.filters.username);
            }

            console.log(`[CivitAI Browser] Loading more results...`);

            const response = await api.fetchApi(`/donut/civitai/search?${params}`);
            if (response.ok) {
                const data = await response.json();
                const newItems = data.items || [];
                const metadata = data.metadata || {};

                // Append new results
                this.searchResults = [...this.searchResults, ...newItems];
                this.currentPage++;
                this.nextCursor = metadata.nextCursor || null;
                this.hasNextPage = !!metadata.nextPage || !!this.nextCursor;

                console.log(`[CivitAI Browser] Loaded ${newItems.length} more, total: ${this.searchResults.length}`);

                // Append new cards to the grid instead of re-rendering
                this.appendToGrid(newItems);
            }
        } catch (error) {
            console.error("[CivitAI Browser] Load more error:", error);
        }

        this.isLoading = false;
        this.updateLoadingState();
    }

    appendToGrid(newItems) {
        const grid = document.getElementById("donut-civitai-grid");
        if (!grid) return;

        for (const model of newItems) {
            grid.appendChild(this.createModelCard(model));
        }

        // Update or add the load more indicator
        this.updateLoadMoreIndicator();
    }

    updateLoadMoreIndicator() {
        const content = this.dialog?.querySelector("#donut-civitai-content");
        if (!content) return;

        // Remove existing indicator
        const existing = content.querySelector(".donut-load-more-indicator");
        if (existing) existing.remove();

        if (this.endlessScroll && this.hasNextPage) {
            const indicator = document.createElement("div");
            indicator.className = "donut-load-more-indicator";
            indicator.style.cssText = `
                text-align: center;
                padding: 20px;
                color: #666;
                font-size: 14px;
            `;
            indicator.textContent = "Scroll down to load more...";
            content.appendChild(indicator);
        }
    }

    setupScrollListener() {
        const content = this.dialog?.querySelector("#donut-civitai-content");
        if (!content) return;

        content.onscroll = () => {
            if (!this.endlessScroll || this.isLoading || !this.hasNextPage) return;

            // Load more when scrolled near bottom (within 200px)
            const scrollBottom = content.scrollHeight - content.scrollTop - content.clientHeight;
            if (scrollBottom < 200) {
                this.loadMore();
            }
        };
    }

    async search(useCursor = null) {
        // If already loading, queue a new search after this one finishes
        if (this.isLoading) {
            this.pendingSearch = true;
            return;
        }

        // Check if the query is a CivitAI URL - if so, load model directly
        const civitaiUrl = this.parseCivitaiUrl(this.filters.query);
        if (civitaiUrl) {
            console.log(`[CivitAI Browser] Detected CivitAI URL, loading model ${civitaiUrl.modelId}`);
            // Clear the search box and load the model details
            this.filters.query = "";
            const searchInput = this.dialog?.querySelector("#donut-civitai-search");
            if (searchInput) searchInput.value = "";
            this.saveFilters();
            // Load model details, optionally selecting a specific version
            this.loadModelDetails(civitaiUrl.modelId, civitaiUrl.versionId);
            return;
        }

        this.isLoading = true;
        this.pendingSearch = false;
        this.updateLoadingState();

        // Save filters whenever we search (but not cursor state)
        this.saveFilters();

        // Use local search if downloadedOnly is enabled
        if (this.filters.downloadedOnly) {
            await this.searchDownloaded();
            this.isLoading = false;
            this.updateLoadingState();
            return;
        }

        try {
            const params = new URLSearchParams({
                query: this.filters.query,
                sort: this.filters.sort,
                period: this.filters.period,
                nsfw: this.filters.nsfw.toString(),
                limit: "20",
                page: this.currentPage.toString()
            });

            // Use cursor if provided (for next/previous page navigation)
            if (useCursor) {
                params.set("cursor", useCursor);
            }

            // Only add types if some are selected
            // CivitAI expects repeated params: types=LORA&types=LoCon
            if (this.filters.types.length > 0) {
                for (const type of this.filters.types) {
                    params.append("types", type);
                }
            }

            // CivitAI expects repeated params for baseModels too
            // Convert display names to API values before sending
            if (this.filters.baseModels.length > 0) {
                for (const baseModel of this.filters.baseModels) {
                    const apiValue = this.displayToApiBaseModel(baseModel);
                    params.append("baseModels", apiValue);
                }
            }

            if (this.filters.tag) {
                params.set("tag", this.filters.tag);
            }

            if (this.filters.username) {
                params.set("username", this.filters.username);
            }

            console.log(`[CivitAI Browser] Searching page ${this.currentPage}, cursor: ${useCursor ? useCursor.substring(0, 20) + '...' : 'none'}`);
            console.log(`[CivitAI Browser] Filters - types: [${this.filters.types.join(', ')}], baseModels (display): [${this.filters.baseModels.join(', ')}]`);
            console.log(`[CivitAI Browser] Full URL params: ${params.toString()}`);

            const response = await api.fetchApi(`/donut/civitai/search?${params}`);
            if (response.ok) {
                const data = await response.json();
                this.searchResults = data.items || [];
                const metadata = data.metadata || {};

                // CivitAI uses cursor pagination - store next cursor
                this.nextCursor = metadata.nextCursor || null;
                this.hasNextPage = !!metadata.nextPage || !!this.nextCursor;

                console.log(`[CivitAI Browser] Got ${this.searchResults.length} results, hasNext: ${this.hasNextPage}`);
                this.renderGrid();
            } else {
                console.error("[CivitAI Browser] Search failed:", await response.text());
                this.searchResults = [];
                this.renderGrid();
            }
        } catch (error) {
            console.error("[CivitAI Browser] Search error:", error);
            this.searchResults = [];
            this.renderGrid();
        }

        this.isLoading = false;
        this.updateLoadingState();

        // If another search was queued while we were loading, run it now
        if (this.pendingSearch) {
            this.pendingSearch = false;
            this.search();
        }
    }

    async searchDownloaded() {
        // Search locally downloaded LoRAs using cached CivitAI metadata
        try {
            // Get list of local LoRAs with their cached metadata
            const response = await api.fetchApi("/donut/loras/list?include_meta=true");
            if (!response.ok) {
                console.error("[CivitAI Browser] Failed to get local LoRAs");
                this.searchResults = [];
                this.renderGrid();
                return;
            }

            const data = await response.json();
            const localLoras = data.loras || [];

            console.log(`[CivitAI Browser] Found ${localLoras.length} local LoRAs`);

            // Filter and transform to CivitAI-like model format
            const results = [];
            const query = this.filters.query.toLowerCase();

            for (const lora of localLoras) {
                // Apply text search filter
                if (query) {
                    const matchName = lora.name?.toLowerCase().includes(query);
                    const matchCivitai = lora.civitai_name?.toLowerCase().includes(query);
                    if (!matchName && !matchCivitai) continue;
                }

                // Apply base model filter
                if (this.filters.baseModels.length > 0 && lora.base_model) {
                    // Convert filter display names to check against stored base_model
                    const matchesBase = this.filters.baseModels.some(bm => {
                        const apiVal = this.displayToApiBaseModel(bm);
                        return lora.base_model === apiVal || lora.base_model === bm;
                    });
                    if (!matchesBase) continue;
                }

                // Check if this model has an update available
                const updateInfo = lora.model_id ? this.modelsWithUpdates.get(lora.model_id) : null;

                // Apply updates filter
                if (this.filters.updatesOnly && !updateInfo) {
                    continue;
                }

                // Create a CivitAI-like model object for display
                // Use first 10 chars of hash for preview URL (matches cache format)
                const hashPrefix = lora.hash ? lora.hash.substring(0, 10) : null;
                // Use full sha256 for deletion (lora.sha256 is the full hash)
                const fullSha256 = lora.sha256 || lora.hash;
                // Use api.apiURL to get the proper server URL for the preview
                const previewUrl = hashPrefix ? api.apiURL(`/donut/loras/preview?hash=${hashPrefix}&type=collage`) : null;

                results.push({
                    id: fullSha256 || lora.name,  // Use full hash as ID for local items
                    name: lora.civitai_name || lora.filename.replace(/\.[^.]+$/, ''),
                    type: "LORA",
                    nsfw: false,
                    modelVersions: [{
                        name: lora.civitai_version || "Local",
                        baseModel: lora.base_model || "",
                        files: [{
                            name: lora.filename,
                            hashes: { SHA256: fullSha256?.toUpperCase() }
                        }],
                        // Add preview image in CivitAI format
                        images: previewUrl ? [{ url: previewUrl, type: "image" }] : []
                    }],
                    stats: {},
                    // Extra data for local items
                    _isLocal: true,
                    _localPath: lora.full_path,
                    _relativePath: lora.name,
                    _modelId: lora.model_id,
                    // Update info if available
                    _hasUpdate: !!updateInfo,
                    _updateInfo: updateInfo
                });
            }

            console.log(`[CivitAI Browser] Filtered to ${results.length} matching LoRAs`);

            this.searchResults = results;
            this.hasNextPage = false;  // No pagination for local search
            this.nextCursor = null;
            this.renderGrid();

        } catch (error) {
            console.error("[CivitAI Browser] Error searching local LoRAs:", error);
            this.searchResults = [];
            this.renderGrid();
        }
    }

    /**
     * Parse a CivitAI URL to extract model ID and optional version ID
     * Supports formats like:
     *   https://civitai.com/models/2201058/model-name
     *   https://civitai.com/models/2201058/model-name?modelVersionId=2483834
     *   https://civitai.com/models/2201058?modelVersionId=2483834
     * @returns {object|null} { modelId, versionId } or null if not a CivitAI URL
     */
    parseCivitaiUrl(text) {
        if (!text) return null;

        // Match civitai.com/models/{id} pattern
        const urlPattern = /civitai\.com\/models\/(\d+)/i;
        const match = text.match(urlPattern);

        if (!match) return null;

        const modelId = parseInt(match[1], 10);
        let versionId = null;

        // Try to extract modelVersionId from query params
        const versionPattern = /modelVersionId=(\d+)/i;
        const versionMatch = text.match(versionPattern);
        if (versionMatch) {
            versionId = parseInt(versionMatch[1], 10);
        }

        return { modelId, versionId };
    }

    async loadModelDetails(modelId, preselectedVersionId = null) {
        this.isLoading = true;
        this.updateLoadingState();

        // Save scroll position before navigating to detail view
        const content = this.dialog?.querySelector("#donut-civitai-content");
        if (content) {
            this.savedScrollPosition = content.scrollTop;
        }

        try {
            const response = await api.fetchApi(`/donut/civitai/model/${modelId}`);
            if (response.ok) {
                this.currentModel = await response.json();
                // Store the preselected version ID for renderDetailView to use
                this.currentModel._preselectedVersionId = preselectedVersionId;
                this.currentView = "detail";
                this.renderDetailView();
            }
        } catch (error) {
            console.error("[CivitAI Browser] Error loading model:", error);
        }

        this.isLoading = false;
        this.updateLoadingState();
    }

    async startDownload(modelVersion, loadToSlot = null, buttonElement = null) {
        const model = this.currentModel;
        if (!model || !modelVersion) return;

        const file = modelVersion.files?.[0];
        if (!file) {
            alert("No downloadable file found");
            return;
        }

        // Update button to show downloading state
        if (buttonElement) {
            buttonElement.disabled = true;
            buttonElement.dataset.originalText = buttonElement.innerHTML;
            buttonElement.dataset.originalBg = buttonElement.style.background;
            buttonElement.innerHTML = `<span class="donut-dl-text">Starting...</span>`;
            buttonElement.style.background = "#555";
            buttonElement.style.position = "relative";
            buttonElement.style.overflow = "hidden";

            // Add progress bar overlay
            const progressBar = document.createElement("div");
            progressBar.className = "donut-dl-progress";
            progressBar.style.cssText = `
                position: absolute;
                left: 0;
                top: 0;
                height: 100%;
                width: 0%;
                background: rgba(90, 140, 90, 0.5);
                transition: width 0.3s;
                z-index: 0;
            `;
            buttonElement.insertBefore(progressBar, buttonElement.firstChild);

            // Make text appear above progress bar
            const textSpan = buttonElement.querySelector(".donut-dl-text");
            if (textSpan) textSpan.style.cssText = "position: relative; z-index: 1;";
        }

        try {
            const response = await api.fetchApi("/donut/civitai/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    downloadUrl: file.downloadUrl,
                    modelType: model.type,
                    baseModel: modelVersion.baseModel,
                    filename: file.name,
                    sha256: file.hashes?.SHA256  // Pass hash to save after download
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.activeDownloads[data.downloadId] = {
                    id: data.downloadId,
                    name: model.name,
                    filename: file.name,
                    savePath: data.savePath,
                    modelType: model.type,
                    loadToSlot: loadToSlot,  // Slot to load into after download (1, 2, or 3)
                    targetNode: this.targetNode,  // The node to load into
                    buttonElement: buttonElement,  // Track button for progress updates
                    sha256: file.hashes?.SHA256  // Store hash for adding to localHashes on completion
                };
                this.startDownloadPolling(data.downloadId);

                // Show "download started" notification
                this.showNotification("⬇ Download Started", file.name, 3000);

                // Show downloads panel automatically when a download starts
                this.showDownloadsPanel();
                this.renderDownloadsPanel();

                // If loading to slot, close the browser
                if (loadToSlot !== null) {
                    this.close();
                }
            } else {
                const error = await response.json();
                // Reset button on error
                if (buttonElement) {
                    this.resetDownloadButton(buttonElement, "Error!");
                    setTimeout(() => this.resetDownloadButton(buttonElement), 2000);
                }

                // Handle duplicate file error specially
                if (error.error === "duplicate") {
                    this.showNotification(
                        "Already Downloaded",
                        `File exists: ${error.existingFile}`,
                        5000
                    );
                    // Add to local hashes so badge shows
                    if (file.hashes?.SHA256) {
                        this.addLocalHash(file.hashes.SHA256);
                    }
                } else {
                    alert(`Download failed: ${error.error || error.message}`);
                }
            }
        } catch (error) {
            console.error("[CivitAI Browser] Download error:", error);
            // Reset button on error
            if (buttonElement) {
                this.resetDownloadButton(buttonElement, "Error!");
                setTimeout(() => this.resetDownloadButton(buttonElement), 2000);
            }
            alert(`Download error: ${error.message}`);
        }
    }

    resetDownloadButton(buttonElement, text = null) {
        if (!buttonElement) return;
        const progressBar = buttonElement.querySelector(".donut-dl-progress");
        if (progressBar) progressBar.remove();

        buttonElement.disabled = false;
        buttonElement.innerHTML = text || buttonElement.dataset.originalText || "Download";
        buttonElement.style.background = buttonElement.dataset.originalBg || "#4a6fa5";
    }

    updateDownloadButton(downloadId, progress, status) {
        const downloadInfo = this.activeDownloads[downloadId];
        if (!downloadInfo?.buttonElement) return;

        const btn = downloadInfo.buttonElement;
        const progressBar = btn.querySelector(".donut-dl-progress");
        const textSpan = btn.querySelector(".donut-dl-text");

        if (status === "downloading") {
            if (progressBar) progressBar.style.width = `${progress}%`;
            if (textSpan) textSpan.textContent = `${Math.round(progress)}%`;
        } else if (status === "completed") {
            if (progressBar) {
                progressBar.style.width = "100%";
                progressBar.style.background = "rgba(90, 160, 90, 0.7)";
            }
            if (textSpan) textSpan.textContent = "Downloaded!";
            btn.style.background = "#4a8a4a";
            // Keep the completed state visible
        } else if (status === "error") {
            this.resetDownloadButton(btn, "Error!");
            setTimeout(() => this.resetDownloadButton(btn), 3000);
        }
    }

    startDownloadPolling(downloadId) {
        const poll = async () => {
            if (!this.activeDownloads[downloadId]) return;

            try {
                const response = await api.fetchApi(`/donut/civitai/download/status/${downloadId}`);
                if (response.ok) {
                    const status = await response.json();
                    const downloadInfo = this.activeDownloads[downloadId];
                    this.activeDownloads[downloadId] = {
                        ...downloadInfo,
                        ...status
                    };
                    this.renderDownloadsPanel();

                    // Update button progress
                    this.updateDownloadButton(downloadId, status.progress || 0, status.status);

                    if (status.status === "downloading") {
                        setTimeout(poll, 1000);
                    } else if (status.status === "completed") {
                        // Add downloaded hash to local hashes so "Downloaded" badge shows immediately
                        if (downloadInfo.sha256) {
                            this.addLocalHash(downloadInfo.sha256);
                        }

                        // Refresh folder cache so the file appears in dropdowns
                        await this.refreshFolderCache(downloadInfo.modelType);

                        // If we need to load to a slot, do it
                        if (downloadInfo.loadToSlot != null && downloadInfo.targetNode) {
                            await this.loadToNodeSlot(
                                downloadInfo.targetNode,
                                downloadInfo.loadToSlot,
                                status.filepath
                            );
                        }

                        // Show completion notification
                        this.showDownloadCompleteNotification(
                            status.filename,
                            status.filepath,
                            downloadInfo.loadToSlot,
                            downloadInfo.targetNode,
                            downloadInfo.modelType
                        );
                    } else if (status.status === "error") {
                        // Update button to show error
                        this.updateDownloadButton(downloadId, 0, "error");
                    }
                }
            } catch (error) {
                console.error("[CivitAI Browser] Poll error:", error);
            }
        };

        poll();
    }

    async refreshFolderCache(modelType) {
        // Refresh the folder cache so newly downloaded files appear in dropdowns
        try {
            // Map model type to folder name
            const folderMap = {
                "LORA": "loras",
                "LoCon": "loras",
                "DoRA": "loras",
                "Checkpoint": "checkpoints",
                "TextualInversion": "embeddings",
                "Controlnet": "controlnet",
                "Upscaler": "upscale_models",
                "VAE": "vae"
            };
            const folder = folderMap[modelType] || "loras";

            await api.fetchApi("/donut/refresh_folder_cache", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ folder })
            });
            console.log(`[CivitAI Browser] Refreshed ${folder} cache`);

            // Also refresh ComfyUI's node definitions so dropdowns update
            if (window.app && typeof window.app.refreshComboInNodes === "function") {
                await window.app.refreshComboInNodes();
                console.log(`[CivitAI Browser] Refreshed node combo widgets`);
            }
        } catch (error) {
            console.error("[CivitAI Browser] Error refreshing cache:", error);
        }
    }

    async loadToNodeSlot(node, slot, filepath) {
        // Load a downloaded LoRA into a specific slot on a DonutLoRAStack node
        if (!node || slot < 1 || slot > 3) return;

        try {
            // Get the relative filename from the full path
            const response = await api.fetchApi(`/donut/loras/filename?path=${encodeURIComponent(filepath)}`);
            if (!response.ok) {
                console.error("[CivitAI Browser] Error getting filename:", await response.text());
                return;
            }
            const data = await response.json();
            const loraFilename = data.filename;

            console.log(`[CivitAI Browser] Looking for lora_name_${slot} widget to set to: ${loraFilename}`);

            // Find the correct widget by name (lora_name_1, lora_name_2, lora_name_3)
            const widgetName = `lora_name_${slot}`;
            const widget = node.widgets?.find(w => w.name === widgetName);

            if (widget) {
                // Also enable the switch for this slot
                const switchWidget = node.widgets?.find(w => w.name === `switch_${slot}`);
                if (switchWidget) {
                    switchWidget.value = "On";
                }

                // Add the new file to widget options if not already present
                // This ensures the dropdown includes the newly downloaded file
                if (widget.options?.values && Array.isArray(widget.options.values)) {
                    if (!widget.options.values.includes(loraFilename)) {
                        widget.options.values.push(loraFilename);
                        widget.options.values.sort();
                        console.log(`[CivitAI Browser] Added ${loraFilename} to widget options`);
                    }
                }

                // Set the LoRA name
                widget.value = loraFilename;

                // Trigger widget callback if it exists
                if (widget.callback) {
                    widget.callback(loraFilename);
                }

                node.setDirtyCanvas(true, true);
                console.log(`[CivitAI Browser] Loaded ${loraFilename} into slot ${slot}`);
            } else {
                console.error(`[CivitAI Browser] Could not find widget ${widgetName}. Available widgets:`, node.widgets?.map(w => w.name));
            }
        } catch (error) {
            console.error("[CivitAI Browser] Error loading to slot:", error);
        }
    }

    async loadDownloadedToSlot(sha256, slot, modelName, downloadFallback = null) {
        // Load an already-downloaded LoRA to a slot by finding it via hash
        // If the file was deleted but hash cache is stale, fall back to downloading
        if (!this.targetNode || !sha256) return false;

        try {
            const response = await api.fetchApi(`/donut/loras/by-hash?sha256=${encodeURIComponent(sha256)}`);
            if (!response.ok) {
                console.error("[CivitAI Browser] Error finding LoRA by hash");
                // Fall back to download if provided
                if (downloadFallback) {
                    console.log("[CivitAI Browser] Falling back to download");
                    this.downloadAndLoadToSlot(
                        downloadFallback.url,
                        downloadFallback.filename,
                        downloadFallback.modelType,
                        downloadFallback.baseModel,
                        slot,
                        modelName,
                        sha256
                    );
                }
                return false;
            }

            const data = await response.json();
            if (!data.found) {
                console.log("[CivitAI Browser] LoRA not found locally by hash (file may have been deleted)");
                // Remove stale hash from local set
                this.localHashes.delete(sha256.toUpperCase());
                // Fall back to download if provided
                if (downloadFallback) {
                    console.log("[CivitAI Browser] Falling back to download");
                    this.downloadAndLoadToSlot(
                        downloadFallback.url,
                        downloadFallback.filename,
                        downloadFallback.modelType,
                        downloadFallback.baseModel,
                        slot,
                        modelName,
                        sha256
                    );
                } else {
                    this.showNotification("File not found", "The cached file may have been deleted. Please re-download.", 5000);
                }
                return false;
            }

            // Load directly to slot using the filename
            const node = this.targetNode;
            const widgetName = `lora_name_${slot}`;
            const widget = node.widgets?.find(w => w.name === widgetName);

            if (widget) {
                // Enable the switch
                const switchWidget = node.widgets?.find(w => w.name === `switch_${slot}`);
                if (switchWidget) {
                    switchWidget.value = "On";
                }

                // Add the file to widget options if not already present
                if (widget.options?.values && Array.isArray(widget.options.values)) {
                    if (!widget.options.values.includes(data.filename)) {
                        widget.options.values.push(data.filename);
                        widget.options.values.sort();
                    }
                }

                // Set the LoRA
                widget.value = data.filename;
                if (widget.callback) {
                    widget.callback(data.filename);
                }
                node.setDirtyCanvas(true, true);

                this.showNotification("✓ Loaded to Slot " + slot, modelName || data.filename, 3000);
                this.close();
                return true;
            }
        } catch (error) {
            console.error("[CivitAI Browser] Error loading downloaded LoRA:", error);
        }
        return false;
    }

    async deleteLoraAndRefresh(sha256, modelName, cardElement, localPath = null) {
        // Delete a LoRA file by its hash (or path) and remove it from the current view
        if (!sha256 && !localPath) {
            this.showNotification("Error", "No hash or path available for this file", 3000);
            return false;
        }

        // Confirm deletion
        if (!confirm(`Delete "${modelName}"?\n\nThis will permanently remove the LoRA file and its cached preview from disk.`)) {
            return false;
        }

        try {
            let response;
            if (sha256) {
                // Delete by hash (preferred)
                response = await api.fetchApi(`/donut/loras/by-hash?sha256=${encodeURIComponent(sha256)}`, {
                    method: "DELETE"
                });
            } else {
                // Delete by path (fallback for files without cached hash)
                response = await api.fetchApi(`/donut/loras/by-path?path=${encodeURIComponent(localPath)}`, {
                    method: "DELETE"
                });
            }

            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                this.showNotification("Delete failed", errData.error || "Unknown error", 5000);
                return false;
            }

            const data = await response.json();

            // Remove from local hashes set if we have a hash
            if (sha256) {
                this.localHashes.delete(sha256.toUpperCase());
            }

            // Remove the card from the DOM immediately
            if (cardElement && cardElement.parentNode) {
                cardElement.style.transition = "opacity 0.3s, transform 0.3s";
                cardElement.style.opacity = "0";
                cardElement.style.transform = "scale(0.9)";
                setTimeout(() => {
                    cardElement.remove();
                }, 300);
            }

            // Remove from searchResults array
            this.searchResults = this.searchResults.filter(m => {
                if (sha256) {
                    const mHash = m.modelVersions?.[0]?.files?.[0]?.hashes?.SHA256;
                    return mHash?.toUpperCase() !== sha256.toUpperCase();
                } else {
                    return m._localPath !== localPath;
                }
            });

            this.showNotification("✓ Deleted", modelName, 3000);
            return true;
        } catch (error) {
            console.error("[CivitAI Browser] Error deleting LoRA:", error);
            this.showNotification("Delete failed", error.message || "Unknown error", 5000);
            return false;
        }
    }

    async fetchMissingInfo() {
        // Fetch CivitAI metadata for local LoRAs that don't have cached info
        if (this.isFetchingMissing) {
            // Already fetching - abort
            this.fetchMissingAborted = true;
            return;
        }

        // Get list of all local LoRAs with metadata
        let allLoras;
        try {
            const response = await api.fetchApi("/donut/loras/list?include_meta=true");
            if (response.ok) {
                const data = await response.json();
                allLoras = data.loras || [];
            } else {
                this.showNotification("Error", "Failed to load LoRA list", 3000);
                return;
            }
        } catch (error) {
            console.error("[CivitAI Browser] Error loading LoRA list:", error);
            this.showNotification("Error", "Failed to load LoRA list", 3000);
            return;
        }

        // Filter to LoRAs without preview (missing CivitAI info)
        const lorasToFetch = allLoras.filter(l => !l.has_preview);

        if (lorasToFetch.length === 0) {
            this.showNotification("All done!", "All LoRAs already have CivitAI info cached", 3000);
            return;
        }

        this.isFetchingMissing = true;
        this.fetchMissingAborted = false;

        // Update button to show Stop state
        const fetchBtn = this.dialog?.querySelector("#donut-civitai-fetch-missing");
        if (fetchBtn) {
            fetchBtn.textContent = "Stop";
            fetchBtn.style.background = "#4a2a2a";
            fetchBtn.style.borderColor = "#6a3a3a";
            fetchBtn.style.color = "#faa";
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

        // Process in batches
        for (let i = 0; i < lorasToFetch.length; i += PARALLEL) {
            if (this.fetchMissingAborted) {
                break;
            }

            const batch = lorasToFetch.slice(i, i + PARALLEL);

            // Update button to show progress
            if (fetchBtn) {
                fetchBtn.textContent = `Stop (${Math.min(i + PARALLEL, total)}/${total})`;
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

                            // Update localHashes with the new hash
                            if (data.hash) {
                                const fullSha256 = data.civitai?.files?.[0]?.hashes?.SHA256;
                                if (fullSha256) {
                                    this.localHashes.add(fullSha256.toUpperCase());
                                }
                            }
                            return "found";
                        } else {
                            return "notFound";
                        }
                    } else {
                        return "error";
                    }
                } catch (e) {
                    console.error(`[CivitAI Browser] Error fetching info for ${lora.name}:`, e);
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
            if (!this.fetchMissingAborted && i + PARALLEL < lorasToFetch.length) {
                await new Promise(resolve => setTimeout(resolve, DELAY_MS));
            }
        }

        // Done fetching
        this.isFetchingMissing = false;

        // Reset button
        if (fetchBtn) {
            fetchBtn.textContent = "Fetch Missing Info";
            fetchBtn.style.background = "#2a4a3a";
            fetchBtn.style.borderColor = "#3a6a4a";
            fetchBtn.style.color = "#8f8";
        }

        // Show results
        if (this.fetchMissingAborted) {
            this.showNotification("Stopped", `Found: ${found}, Not on CivitAI: ${notFound}, Errors: ${errors}`, 5000);
        } else {
            this.showNotification("Done!", `Found: ${found}, Not on CivitAI: ${notFound}, Errors: ${errors}`, 5000);
        }

        this.fetchMissingAborted = false;

        // If in downloaded-only mode, refresh the search to show updated previews
        if (this.filters.downloadedOnly) {
            this.search();
        }
    }

    async checkForUpdates() {
        // Check CivitAI for newer versions of downloaded models
        if (this.isCheckingUpdates) {
            // Already checking - abort
            this.checkUpdatesAborted = true;
            return;
        }

        // Get list of all local LoRAs with metadata
        let allLoras;
        try {
            const response = await api.fetchApi("/donut/loras/list?include_meta=true");
            if (response.ok) {
                const data = await response.json();
                allLoras = data.loras || [];
            } else {
                this.showNotification("Error", "Failed to load LoRA list", 3000);
                return;
            }
        } catch (error) {
            console.error("[CivitAI Browser] Error loading LoRA list:", error);
            this.showNotification("Error", "Failed to load LoRA list", 3000);
            return;
        }

        // Filter to LoRAs that have model_id (from CivitAI)
        const lorasToCheck = allLoras.filter(l => l.model_id && l.sha256);

        if (lorasToCheck.length === 0) {
            this.showNotification("No models to check", "No LoRAs with CivitAI info found. Try 'Fetch Missing Info' first.", 4000);
            return;
        }

        // Collect ALL local hashes - used to check if we already have a newer version
        // under a different filename (e.g., lora_v2.safetensors)
        const allLocalHashes = allLoras
            .filter(l => l.sha256)
            .map(l => l.sha256);

        this.isCheckingUpdates = true;
        this.checkUpdatesAborted = false;

        // Update button to show checking state
        const checkBtn = this.dialog?.querySelector("#donut-civitai-check-updates");
        if (checkBtn) {
            checkBtn.textContent = "Checking...";
            checkBtn.style.background = "#5a4a2a";
        }

        // Clear previous results
        this.modelsWithUpdates.clear();

        // Process in batches to avoid rate limiting (5 models per request is reasonable)
        const BATCH_SIZE = 5;
        const DELAY_MS = 1500;  // 1.5 seconds between batches for rate limiting
        let checked = 0;
        let updatesFound = 0;

        for (let i = 0; i < lorasToCheck.length; i += BATCH_SIZE) {
            if (this.checkUpdatesAborted) {
                break;
            }

            const batch = lorasToCheck.slice(i, i + BATCH_SIZE);

            // Update button to show progress
            if (checkBtn) {
                checkBtn.textContent = `Checking (${Math.min(i + BATCH_SIZE, lorasToCheck.length)}/${lorasToCheck.length})`;
            }

            try {
                const response = await api.fetchApi("/donut/civitai/check-updates", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        models: batch.map(l => ({
                            model_id: l.model_id,
                            sha256: l.sha256
                        })),
                        // Send all local hashes so server can check if we already have the latest
                        all_local_hashes: allLocalHashes
                    })
                });

                if (response.ok) {
                    const data = await response.json();
                    const updates = data.updates || [];

                    for (const update of updates) {
                        this.modelsWithUpdates.set(update.model_id, update);
                        updatesFound++;
                    }
                }
            } catch (error) {
                console.error("[CivitAI Browser] Error checking updates:", error);
            }

            checked += batch.length;

            // Rate limiting delay between batches
            if (!this.checkUpdatesAborted && i + BATCH_SIZE < lorasToCheck.length) {
                await new Promise(resolve => setTimeout(resolve, DELAY_MS));
            }
        }

        this.isCheckingUpdates = false;

        // Reset button
        if (checkBtn) {
            checkBtn.textContent = "Check for Updates";
            checkBtn.style.background = "#4a3a2a";
        }

        // Show results
        if (this.checkUpdatesAborted) {
            this.showNotification("Stopped", `Found ${updatesFound} updates (checked ${checked}/${lorasToCheck.length})`, 5000);
        } else {
            this.showNotification("Check Complete", `Found ${updatesFound} LoRAs with updates available`, 5000);
        }

        this.checkUpdatesAborted = false;

        // Enable the updates filter checkbox if updates found
        const updatesCheckbox = this.dialog?.querySelector("#donut-civitai-updates-checkbox");
        if (updatesCheckbox && updatesFound > 0) {
            updatesCheckbox.disabled = false;
        }

        // If in downloaded-only mode, refresh the search to show update badges
        if (this.filters.downloadedOnly) {
            this.search();
        }
    }

    showNotification(title, message, duration = 5000) {
        // Create toast notification
        const toast = document.createElement("div");
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #2a4a3a;
            color: #fff;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10001;
            animation: slideIn 0.3s ease;
        `;
        toast.innerHTML = `
            <div style="font-weight: bold; margin-bottom: 5px;">${title}</div>
            <div style="font-size: 11px; color: #aaa; word-break: break-all;">${message}</div>
        `;
        document.body.appendChild(toast);

        if (duration > 0) {
            setTimeout(() => {
                toast.style.animation = "slideOut 0.3s ease";
                setTimeout(() => toast.remove(), 300);
            }, duration);
        }

        return toast;
    }

    showDownloadCompleteNotification(filename, filepath, loadedToSlot, targetNode, modelType) {
        // Create a more interactive notification for completed downloads
        const toast = document.createElement("div");
        toast.style.cssText = `
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #2a4a3a;
            color: #fff;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            z-index: 10001;
            animation: slideIn 0.3s ease;
            max-width: 350px;
        `;

        let content = `
            <div style="font-weight: bold; margin-bottom: 5px;">✓ Downloaded: ${filename}</div>
            <div style="font-size: 11px; color: #aaa; margin-bottom: 10px; word-break: break-all;">${filepath}</div>
        `;

        // If loaded to a slot, show confirmation
        if (loadedToSlot !== null && targetNode) {
            content += `<div style="font-size: 12px; color: #8f8;">Loaded to LoRA Slot ${loadedToSlot}</div>`;
        }
        // If it's a LoRA type and we have no target, offer to load to any node
        else if (modelType && ["LORA", "LoCon", "DoRA"].includes(modelType)) {
            content += `<div style="font-size: 11px; color: #888; margin-bottom: 8px;">Click a slot to load into a LoRA node:</div>`;
            content += `<div style="display: flex; gap: 8px;">`;
            for (let i = 1; i <= 3; i++) {
                content += `<button class="load-slot-btn" data-slot="${i}" style="
                    flex: 1;
                    padding: 6px 12px;
                    background: #4a6fa5;
                    border: none;
                    border-radius: 4px;
                    color: white;
                    font-size: 12px;
                    cursor: pointer;
                ">Slot ${i}</button>`;
            }
            content += `</div>`;
        }

        toast.innerHTML = content;

        // Add click handlers for slot buttons
        toast.querySelectorAll(".load-slot-btn").forEach(btn => {
            btn.onclick = async () => {
                const slot = parseInt(btn.dataset.slot);
                // Find a DonutLoRAStack node to load into
                const loraNodes = app.graph._nodes.filter(n =>
                    n.type === "DonutLoRAStack" || n.comfyClass === "DonutLoRAStack"
                );
                if (loraNodes.length > 0) {
                    // Use the first selected LoRA node, or the first one if none selected
                    const selectedNode = loraNodes.find(n => n.is_selected) || loraNodes[0];
                    await this.loadToNodeSlot(selectedNode, slot, filepath);
                    btn.textContent = "✓";
                    btn.style.background = "#4a8a4a";
                    setTimeout(() => {
                        toast.style.animation = "slideOut 0.3s ease";
                        setTimeout(() => toast.remove(), 300);
                    }, 1000);
                } else {
                    alert("No DonutLoRAStack node found in the workflow");
                }
            };
        });

        // Close button
        const closeBtn = document.createElement("div");
        closeBtn.innerHTML = "×";
        closeBtn.style.cssText = `
            position: absolute;
            top: 5px;
            right: 10px;
            cursor: pointer;
            font-size: 18px;
            color: #888;
        `;
        closeBtn.onclick = () => {
            toast.style.animation = "slideOut 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        };
        toast.style.position = "fixed";
        toast.appendChild(closeBtn);

        document.body.appendChild(toast);

        // Auto-dismiss after 15 seconds if no action taken
        setTimeout(() => {
            if (toast.parentNode) {
                toast.style.animation = "slideOut 0.3s ease";
                setTimeout(() => toast.remove(), 300);
            }
        }, 15000);
    }

    createDialog() {
        const overlay = document.createElement("div");
        overlay.id = "donut-civitai-overlay";
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.85);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        const dialog = document.createElement("div");
        dialog.id = "donut-civitai-dialog";
        dialog.style.cssText = `
            background: #1a1a2e;
            border-radius: 12px;
            width: 95vw;
            max-width: 1400px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
            color: #eee;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            overflow: hidden;
        `;

        // Header
        dialog.appendChild(this.createHeader());

        // Main content area
        const main = document.createElement("div");
        main.id = "donut-civitai-main";
        main.style.cssText = `
            flex: 1;
            display: flex;
            overflow: hidden;
        `;

        // Sidebar with filters
        main.appendChild(this.createSidebar());

        // Content area (grid or detail view)
        const content = document.createElement("div");
        content.id = "donut-civitai-content";
        content.style.cssText = `
            flex: 1;
            overflow-y: auto;
            padding: 20px;
        `;
        main.appendChild(content);

        dialog.appendChild(main);

        // Downloads panel (hidden by default)
        dialog.appendChild(this.createDownloadsPanel());

        overlay.appendChild(dialog);

        // Close on overlay click
        overlay.onclick = (e) => {
            if (e.target === overlay) this.close();
        };

        // Keyboard shortcuts
        overlay.tabIndex = 0;
        overlay.onkeydown = (e) => {
            if (e.key === "Escape") {
                if (this.currentView === "detail") {
                    this.currentView = "grid";
                    this.renderGrid(true);  // Restore scroll position
                } else {
                    this.close();
                }
                e.preventDefault();
            }
        };

        this.dialog = overlay;
        return overlay;
    }

    createHeader() {
        const header = document.createElement("div");
        header.style.cssText = `
            display: flex;
            align-items: center;
            gap: 15px;
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            background: #151525;
        `;

        // Title
        const title = document.createElement("h2");
        title.innerHTML = `<span style="color: #6a9fd4;">CivitAI</span> Browser`;
        title.style.cssText = "margin: 0; font-size: 18px; font-weight: 600;";
        header.appendChild(title);

        // Search input
        const searchContainer = document.createElement("div");
        searchContainer.style.cssText = `
            flex: 1;
            max-width: 500px;
            position: relative;
        `;

        const searchInput = document.createElement("input");
        searchInput.type = "text";
        searchInput.id = "donut-civitai-search";
        searchInput.placeholder = "Search models...";
        searchInput.value = this.filters.query;
        searchInput.style.cssText = `
            width: 100%;
            padding: 10px 15px;
            padding-left: 40px;
            padding-right: 35px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #eee;
            font-size: 14px;
            box-sizing: border-box;
        `;

        // Clear button
        const clearBtn = document.createElement("span");
        clearBtn.innerHTML = "×";
        clearBtn.id = "donut-civitai-search-clear";
        clearBtn.style.cssText = `
            position: absolute;
            right: 10px;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
            font-size: 20px;
            cursor: pointer;
            display: ${this.filters.query ? 'block' : 'none'};
            line-height: 1;
            padding: 2px 6px;
            border-radius: 50%;
        `;
        clearBtn.onmouseover = () => { clearBtn.style.color = '#fff'; clearBtn.style.background = '#444'; };
        clearBtn.onmouseout = () => { clearBtn.style.color = '#666'; clearBtn.style.background = 'transparent'; };
        clearBtn.onclick = () => {
            searchInput.value = "";
            this.filters.query = "";
            clearBtn.style.display = "none";
            this.resetPagination();
            this.search();
            searchInput.focus();
        };

        searchInput.oninput = (e) => {
            this.filters.query = e.target.value;
            clearBtn.style.display = e.target.value ? "block" : "none";
            clearTimeout(this.searchTimeout);
            this.searchTimeout = setTimeout(() => {
                this.resetPagination();
                this.search();
            }, 300);
        };
        searchInput.onkeydown = (e) => {
            if (e.key === "Enter") {
                clearTimeout(this.searchTimeout);
                this.resetPagination();
                this.search();
            }
            e.stopPropagation();
        };

        // Search icon
        const searchIcon = document.createElement("span");
        searchIcon.innerHTML = "&#128269;";
        searchIcon.style.cssText = `
            position: absolute;
            left: 12px;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
            font-size: 16px;
            pointer-events: none;
        `;

        searchContainer.appendChild(searchIcon);
        searchContainer.appendChild(searchInput);
        searchContainer.appendChild(clearBtn);
        header.appendChild(searchContainer);

        // Refresh button
        const refreshBtn = document.createElement("button");
        refreshBtn.innerHTML = "↻";
        refreshBtn.title = "Refresh results";
        refreshBtn.style.cssText = `
            background: #252538;
            color: #eee;
            border: 1px solid #333;
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 18px;
            cursor: pointer;
            transition: background 0.2s;
        `;
        refreshBtn.onmouseenter = () => refreshBtn.style.background = "#353548";
        refreshBtn.onmouseleave = () => refreshBtn.style.background = "#252538";
        refreshBtn.onclick = () => {
            this.currentPage = 1;
            this.search();
        };
        header.appendChild(refreshBtn);

        // Downloads button
        const downloadsBtn = document.createElement("button");
        downloadsBtn.id = "donut-civitai-downloads-btn";
        downloadsBtn.innerHTML = "&#8595; Downloads";
        downloadsBtn.style.cssText = this.getButtonStyle();
        downloadsBtn.onclick = () => this.toggleDownloadsPanel();
        header.appendChild(downloadsBtn);

        // Close button
        const closeBtn = document.createElement("button");
        closeBtn.innerHTML = "&times;";
        closeBtn.style.cssText = `
            background: none;
            border: none;
            color: #888;
            font-size: 28px;
            cursor: pointer;
            padding: 0 10px;
            line-height: 1;
        `;
        closeBtn.onclick = () => this.close();
        header.appendChild(closeBtn);

        return header;
    }

    createSidebar() {
        const sidebar = document.createElement("div");
        sidebar.id = "donut-civitai-sidebar";
        sidebar.style.cssText = `
            width: 220px;
            min-width: 220px;
            background: #151525;
            padding: 15px;
            overflow-y: auto;
            border-right: 1px solid #333;
        `;

        // Clear All Filters button (only show if filters are active)
        const clearBtnContainer = document.createElement("div");
        clearBtnContainer.id = "donut-civitai-clear-filters";
        clearBtnContainer.style.cssText = "margin-bottom: 15px;";

        if (this.hasActiveFilters()) {
            const clearBtn = document.createElement("button");
            clearBtn.textContent = "✕ Clear All Filters";
            clearBtn.style.cssText = `
                width: 100%;
                padding: 8px 12px;
                background: #6a3a3a;
                border: none;
                border-radius: 6px;
                color: #fff;
                font-size: 12px;
                cursor: pointer;
                transition: background 0.2s;
            `;
            clearBtn.onmouseenter = () => clearBtn.style.background = "#7a4a4a";
            clearBtn.onmouseleave = () => clearBtn.style.background = "#6a3a3a";
            clearBtn.onclick = () => this.clearFilters();
            clearBtnContainer.appendChild(clearBtn);
        }
        sidebar.appendChild(clearBtnContainer);

        // Top toggles section (Downloaded only, NSFW, Endless scroll)
        const togglesContainer = document.createElement("div");
        togglesContainer.style.cssText = "margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid #333;";

        // Downloaded Only toggle
        const downloadedContainer = document.createElement("div");
        downloadedContainer.style.cssText = "margin-bottom: 8px;";

        const downloadedLabel = document.createElement("label");
        downloadedLabel.style.cssText = "display: flex; align-items: center; gap: 8px; cursor: pointer; color: #8f8; font-size: 13px; font-weight: 500;";

        const downloadedCheckbox = document.createElement("input");
        downloadedCheckbox.type = "checkbox";
        downloadedCheckbox.checked = this.filters.downloadedOnly || false;
        downloadedCheckbox.onchange = (e) => {
            this.filters.downloadedOnly = e.target.checked;
            this.resetPagination();
            this.search();
        };

        downloadedLabel.appendChild(downloadedCheckbox);
        downloadedLabel.appendChild(document.createTextNode("Downloaded only"));
        downloadedContainer.appendChild(downloadedLabel);

        // Fetch Missing button (for downloaded-only mode)
        const fetchMissingBtn = document.createElement("button");
        fetchMissingBtn.id = "donut-civitai-fetch-missing";
        fetchMissingBtn.textContent = "Fetch Missing Info";
        fetchMissingBtn.title = "Fetch CivitAI metadata for LoRAs without cached info";
        fetchMissingBtn.style.cssText = `
            width: 100%;
            margin-top: 6px;
            padding: 6px 10px;
            background: #2a4a3a;
            border: 1px solid #3a6a4a;
            border-radius: 4px;
            color: #8f8;
            font-size: 11px;
            cursor: pointer;
        `;
        fetchMissingBtn.onclick = () => this.fetchMissingInfo();
        downloadedContainer.appendChild(fetchMissingBtn);

        togglesContainer.appendChild(downloadedContainer);

        // Updates Available toggle (sub-option of Downloaded only)
        const updatesContainer = document.createElement("div");
        updatesContainer.style.cssText = "margin-bottom: 8px; margin-left: 20px;";

        const updatesLabel = document.createElement("label");
        updatesLabel.style.cssText = "display: flex; align-items: center; gap: 8px; cursor: pointer; color: #f8a; font-size: 13px; font-weight: 500;";

        const updatesCheckbox = document.createElement("input");
        updatesCheckbox.type = "checkbox";
        updatesCheckbox.id = "donut-civitai-updates-checkbox";
        updatesCheckbox.checked = this.filters.updatesOnly || false;
        updatesCheckbox.disabled = !this.filters.downloadedOnly;
        updatesCheckbox.onchange = (e) => {
            this.filters.updatesOnly = e.target.checked;
            this.resetPagination();
            this.search();
        };

        updatesLabel.appendChild(updatesCheckbox);
        updatesLabel.appendChild(document.createTextNode("Updates available"));
        updatesContainer.appendChild(updatesLabel);

        // Check Updates button
        const checkUpdatesBtn = document.createElement("button");
        checkUpdatesBtn.id = "donut-civitai-check-updates";
        checkUpdatesBtn.textContent = "Check for Updates";
        checkUpdatesBtn.title = "Check CivitAI for newer versions of your downloaded LoRAs";
        checkUpdatesBtn.style.cssText = `
            width: 100%;
            margin-top: 6px;
            padding: 6px 10px;
            background: #4a3a2a;
            border: 1px solid #6a4a3a;
            border-radius: 4px;
            color: #f8a;
            font-size: 11px;
            cursor: pointer;
        `;
        checkUpdatesBtn.onclick = () => this.checkForUpdates();
        updatesContainer.appendChild(checkUpdatesBtn);

        togglesContainer.appendChild(updatesContainer);

        // Update downloadedOnly checkbox to enable/disable updates checkbox
        downloadedCheckbox.onchange = (e) => {
            this.filters.downloadedOnly = e.target.checked;
            updatesCheckbox.disabled = !e.target.checked;
            if (!e.target.checked) {
                this.filters.updatesOnly = false;
                updatesCheckbox.checked = false;
            }
            this.resetPagination();
            this.search();
        };

        // NSFW toggle
        const nsfwContainer = document.createElement("div");
        nsfwContainer.style.cssText = "margin-bottom: 8px;";

        const nsfwLabel = document.createElement("label");
        nsfwLabel.style.cssText = "display: flex; align-items: center; gap: 8px; cursor: pointer; color: #aaa; font-size: 13px;";

        const nsfwCheckbox = document.createElement("input");
        nsfwCheckbox.type = "checkbox";
        nsfwCheckbox.checked = this.filters.nsfw;
        nsfwCheckbox.onchange = (e) => {
            this.filters.nsfw = e.target.checked;
            this.resetPagination();
            this.debouncedSearch();
        };

        nsfwLabel.appendChild(nsfwCheckbox);
        nsfwLabel.appendChild(document.createTextNode("Include NSFW"));
        nsfwContainer.appendChild(nsfwLabel);
        togglesContainer.appendChild(nsfwContainer);

        // Endless scroll toggle
        const scrollContainer = document.createElement("div");

        const scrollLabel = document.createElement("label");
        scrollLabel.style.cssText = "display: flex; align-items: center; gap: 8px; cursor: pointer; color: #aaa; font-size: 13px;";

        const scrollCheckbox = document.createElement("input");
        scrollCheckbox.type = "checkbox";
        scrollCheckbox.checked = this.endlessScroll;
        scrollCheckbox.onchange = (e) => {
            this.toggleEndlessScroll();
            // Update checkbox state in case toggle changed it
            scrollCheckbox.checked = this.endlessScroll;
        };

        scrollLabel.appendChild(scrollCheckbox);
        scrollLabel.appendChild(document.createTextNode("Endless scroll"));
        scrollContainer.appendChild(scrollLabel);
        togglesContainer.appendChild(scrollContainer);

        sidebar.appendChild(togglesContainer);

        // Model Type filter
        sidebar.appendChild(this.createFilterSection("Type", this.modelTypes, this.filters.types, true, (selected) => {
            console.log(`[CivitAI Browser] Type filter changed to: [${selected.join(', ')}]`);
            this.filters.types = selected;
            this.resetPagination();
            this.debouncedSearch();
        }));

        // Sort filter
        sidebar.appendChild(this.createSelectFilter("Sort", this.sortOptions, this.filters.sort, (value) => {
            this.filters.sort = value;
            this.resetPagination();
            this.debouncedSearch();
        }));

        // Period filter
        sidebar.appendChild(this.createSelectFilter("Period", this.periodOptions, this.filters.period, (value) => {
            this.filters.period = value;
            this.resetPagination();
            this.debouncedSearch();
        }));

        // Base Model filter
        sidebar.appendChild(this.createFilterSection("Base Model", this.baseModelOptions, this.filters.baseModels, true, (selected) => {
            console.log(`[CivitAI Browser] Base Model filter changed to: [${selected.join(', ')}]`);
            this.filters.baseModels = selected;
            this.resetPagination();
            this.debouncedSearch();
        }));

        // Tag filter input
        sidebar.appendChild(this.createTextFilter("Tag", this.filters.tag, "e.g. anime, character", (value) => {
            this.filters.tag = value;
            this.resetPagination();
            this.search();
        }, "donut-civitai-tag-filter"));

        // Creator/Username filter input
        sidebar.appendChild(this.createTextFilter("Creator", this.filters.username, "Username", (value) => {
            this.filters.username = value;
            this.resetPagination();
            this.search();
        }, "donut-civitai-creator-filter"));

        return sidebar;
    }

    createTextFilter(title, value, placeholder, onChange, inputId = null) {
        const section = document.createElement("div");
        section.style.cssText = "margin-top: 15px; padding-top: 15px; border-top: 1px solid #333;";

        const header = document.createElement("div");
        header.textContent = title;
        header.style.cssText = "font-size: 12px; font-weight: 600; color: #888; text-transform: uppercase; margin-bottom: 8px;";
        section.appendChild(header);

        // Container for input and clear button
        const inputContainer = document.createElement("div");
        inputContainer.style.cssText = "position: relative;";

        const input = document.createElement("input");
        input.type = "text";
        if (inputId) input.id = inputId;
        input.value = value;
        input.placeholder = placeholder;
        input.style.cssText = `
            width: 100%;
            padding: 8px 10px;
            padding-right: 28px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 6px;
            color: #eee;
            font-size: 13px;
            box-sizing: border-box;
        `;

        // Clear button
        const clearBtn = document.createElement("span");
        clearBtn.innerHTML = "×";
        clearBtn.style.cssText = `
            position: absolute;
            right: 8px;
            top: 50%;
            transform: translateY(-50%);
            color: #666;
            font-size: 16px;
            cursor: pointer;
            display: ${value ? 'block' : 'none'};
            line-height: 1;
            padding: 2px 4px;
            border-radius: 50%;
        `;
        clearBtn.onmouseover = () => { clearBtn.style.color = '#fff'; clearBtn.style.background = '#444'; };
        clearBtn.onmouseout = () => { clearBtn.style.color = '#666'; clearBtn.style.background = 'transparent'; };
        clearBtn.onclick = () => {
            input.value = "";
            clearBtn.style.display = "none";
            onChange("");
            input.focus();
        };

        let debounceTimer;
        input.oninput = (e) => {
            clearBtn.style.display = e.target.value ? "block" : "none";
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => onChange(e.target.value), 500);
        };
        input.onkeydown = (e) => {
            if (e.key === "Enter") {
                clearTimeout(debounceTimer);
                onChange(e.target.value);
            }
            e.stopPropagation();
        };

        inputContainer.appendChild(input);
        inputContainer.appendChild(clearBtn);
        section.appendChild(inputContainer);
        return section;
    }

    createFilterSection(title, options, selected, multiSelect, onChange) {
        const section = document.createElement("div");
        section.style.cssText = "margin-bottom: 20px;";

        const header = document.createElement("div");
        header.textContent = title;
        header.style.cssText = "font-size: 12px; font-weight: 600; color: #888; text-transform: uppercase; margin-bottom: 10px;";
        section.appendChild(header);

        const optionsContainer = document.createElement("div");
        optionsContainer.style.cssText = "display: flex; flex-direction: column; gap: 6px;";

        for (const option of options) {
            const label = document.createElement("label");
            label.style.cssText = "display: flex; align-items: center; gap: 8px; cursor: pointer; color: #ccc; font-size: 13px;";

            const checkbox = document.createElement("input");
            checkbox.type = multiSelect ? "checkbox" : "radio";
            checkbox.name = `filter-${title}`;
            checkbox.value = option;
            checkbox.checked = selected.includes(option);
            checkbox.style.cssText = "cursor: pointer;";
            checkbox.onchange = () => {
                if (multiSelect) {
                    const newSelected = [];
                    optionsContainer.querySelectorAll("input:checked").forEach(cb => newSelected.push(cb.value));
                    onChange(newSelected);
                } else {
                    onChange([option]);
                }
            };

            label.appendChild(checkbox);
            label.appendChild(document.createTextNode(option));
            optionsContainer.appendChild(label);
        }

        section.appendChild(optionsContainer);
        return section;
    }

    createSelectFilter(title, options, selected, onChange) {
        const section = document.createElement("div");
        section.style.cssText = "margin-bottom: 15px;";

        const header = document.createElement("div");
        header.textContent = title;
        header.style.cssText = "font-size: 12px; font-weight: 600; color: #888; text-transform: uppercase; margin-bottom: 8px;";
        section.appendChild(header);

        const select = document.createElement("select");
        select.style.cssText = `
            width: 100%;
            padding: 8px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 4px;
            color: #eee;
            font-size: 13px;
            cursor: pointer;
        `;

        for (const option of options) {
            const opt = document.createElement("option");
            opt.value = option;
            opt.textContent = option;
            opt.selected = option === selected;
            select.appendChild(opt);
        }

        select.onchange = (e) => onChange(e.target.value);
        section.appendChild(select);

        return section;
    }

    createDownloadsPanel() {
        const panel = document.createElement("div");
        panel.id = "donut-civitai-downloads";
        panel.style.cssText = `
            display: none;
            position: absolute;
            top: 60px;
            right: 20px;
            width: 350px;
            max-height: 400px;
            background: #1a1a2e;
            border: 1px solid #333;
            border-radius: 8px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            overflow: hidden;
            z-index: 100;
        `;

        const header = document.createElement("div");
        header.style.cssText = "padding: 12px 15px; border-bottom: 1px solid #333; font-weight: 600;";
        header.textContent = "Downloads";
        panel.appendChild(header);

        const content = document.createElement("div");
        content.id = "donut-civitai-downloads-content";
        content.style.cssText = "max-height: 340px; overflow-y: auto; padding: 10px;";
        content.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No active downloads</div>';
        panel.appendChild(content);

        return panel;
    }

    toggleDownloadsPanel() {
        const panel = this.dialog?.querySelector("#donut-civitai-downloads");
        if (panel) {
            panel.style.display = panel.style.display === "none" ? "block" : "none";
        }
    }

    showDownloadsPanel() {
        const panel = this.dialog?.querySelector("#donut-civitai-downloads");
        if (panel) {
            panel.style.display = "block";
            // Flash the panel to draw attention
            panel.style.boxShadow = "0 0 20px rgba(100, 200, 100, 0.8)";
            setTimeout(() => {
                panel.style.boxShadow = "0 4px 20px rgba(0,0,0,0.5)";
            }, 500);
        }
        // Also highlight the downloads button
        const btn = this.dialog?.querySelector("#donut-civitai-downloads-btn");
        if (btn) {
            btn.style.background = "#4a8a4a";
            setTimeout(() => {
                btn.style.background = "";
            }, 1000);
        }
        // Render the panel content
        this.renderDownloadsPanel();
    }

    renderDownloadsPanel() {
        const content = this.dialog?.querySelector("#donut-civitai-downloads-content");
        if (!content) return;

        const downloads = Object.values(this.activeDownloads);
        if (downloads.length === 0) {
            content.innerHTML = '<div style="color: #666; text-align: center; padding: 20px;">No active downloads</div>';
            return;
        }

        content.innerHTML = "";
        for (const dl of downloads) {
            const item = document.createElement("div");
            item.style.cssText = `
                padding: 10px;
                background: #0d0d1a;
                border-radius: 6px;
                margin-bottom: 8px;
            `;

            const progress = dl.progress || 0;
            const speed = dl.speedBps ? this.formatBytes(dl.speedBps) + "/s" : "";
            const size = dl.totalSize ? this.formatBytes(dl.totalSize) : "";

            let statusColor = "#6a9fd4";
            let statusText = dl.status || "pending";
            let errorMsg = "";
            if (dl.status === "completed") {
                statusColor = "#4a8a4a";
                statusText = "Completed";
            } else if (dl.status === "error") {
                statusColor = "#a44";
                statusText = "Error";
                errorMsg = dl.error || "Unknown error";
            } else if (dl.status === "downloading") {
                statusText = "Downloading...";
            }

            item.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-size: 12px; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 200px;">${dl.filename || dl.name}</span>
                    <span style="font-size: 10px; color: ${statusColor};">${statusText}</span>
                </div>
                <div style="height: 4px; background: #333; border-radius: 2px; overflow: hidden;">
                    <div style="height: 100%; width: ${progress}%; background: ${statusColor}; transition: width 0.3s;"></div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-top: 4px; font-size: 10px; color: #666;">
                    <span>${Math.round(progress)}%</span>
                    <span>${speed} ${size ? `/ ${size}` : ""}</span>
                </div>
                ${errorMsg ? `<div style="font-size: 10px; color: #a44; margin-top: 4px; word-break: break-word;">${errorMsg}</div>` : ""}
            `;

            content.appendChild(item);
        }
    }

    renderGrid(restoreScroll = false) {
        const content = this.dialog?.querySelector("#donut-civitai-content");
        if (!content) return;

        content.innerHTML = "";
        this.currentView = "grid";

        if (this.searchResults.length === 0) {
            content.innerHTML = `
                <div style="text-align: center; padding: 60px 20px; color: #666;">
                    <div style="font-size: 48px; margin-bottom: 20px;">&#128269;</div>
                    <div style="font-size: 16px;">No models found</div>
                    <div style="font-size: 13px; margin-top: 10px;">Try different search terms or filters</div>
                </div>
            `;
            return;
        }

        // Grid container - flexbox for stable item positions during endless scroll
        const grid = document.createElement("div");
        grid.id = "donut-civitai-grid";

        grid.style.cssText = `
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            align-items: flex-start;
        `;

        for (const model of this.searchResults) {
            grid.appendChild(this.createModelCard(model));
        }

        content.appendChild(grid);

        // Show pagination or endless scroll indicator based on setting
        if (this.endlessScroll) {
            // Setup scroll listener for endless scroll
            this.setupScrollListener();
            this.updateLoadMoreIndicator();
        } else {
            // Traditional pagination - show if we have results or can go back/forward
            if (this.currentPage > 1 || this.hasNextPage) {
                content.appendChild(this.createPagination());
            }
        }

        // Restore scroll position if returning from detail view, otherwise scroll to top
        if (restoreScroll && this.savedScrollPosition > 0) {
            // Use requestAnimationFrame to ensure DOM is fully rendered before scrolling
            requestAnimationFrame(() => {
                content.scrollTop = this.savedScrollPosition;
            });
        } else {
            content.scrollTop = 0;
        }
    }

    createModelCard(model) {
        const card = document.createElement("div");
        card.className = "donut-civitai-card";
        card.style.cssText = `
            background: #0d0d1a;
            border-radius: 10px;
            overflow: hidden;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
            border: 2px solid transparent;
            width: calc(25% - 12px);
            min-width: 200px;
            box-sizing: border-box;
        `;

        card.onmouseenter = () => {
            card.style.transform = "translateY(-4px)";
            card.style.boxShadow = "0 8px 25px rgba(0,0,0,0.4)";
            card.style.borderColor = "#4a6fa5";
        };
        card.onmouseleave = () => {
            card.style.transform = "translateY(0)";
            card.style.boxShadow = "none";
            card.style.borderColor = "transparent";
        };

        card.onclick = () => {
            // For local items without CivitAI ID, show a simple info notification
            if (model._isLocal && typeof model.id === 'string') {
                this.showNotification("Local LoRA", `${model.name}\n${model._relativePath}`, 3000);
            } else {
                this.loadModelDetails(model.id);
            }
        };

        // Image - use aspect ratio from CivitAI data if available
        const previewImage = model.modelVersions?.[0]?.images?.[0];
        const aspectRatio = previewImage?.width && previewImage?.height
            ? previewImage.width / previewImage.height
            : 1;
        // Constrain aspect ratio to reasonable bounds
        const constrainedRatio = Math.max(0.5, Math.min(2, aspectRatio));

        const imgContainer = document.createElement("div");
        imgContainer.style.cssText = `
            width: 100%;
            aspect-ratio: ${constrainedRatio};
            background: #151525;
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
            min-height: 120px;
            max-height: 350px;
        `;

        if (previewImage?.url) {
            if (previewImage.type === "video") {
                // Video preview
                const video = document.createElement("video");
                video.src = previewImage.url;
                video.style.cssText = "width: 100%; height: 100%; object-fit: cover;";
                video.muted = true;
                video.loop = true;
                video.playsInline = true;
                // Play on hover
                imgContainer.onmouseenter = () => video.play();
                imgContainer.onmouseleave = () => { video.pause(); video.currentTime = 0; };
                // Video play icon overlay
                const playIcon = document.createElement("div");
                playIcon.innerHTML = "▶";
                playIcon.style.cssText = `
                    position: absolute;
                    top: 50%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: white;
                    font-size: 24px;
                    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
                    pointer-events: none;
                    opacity: 0.8;
                `;
                imgContainer.style.position = "relative";
                imgContainer.appendChild(video);
                imgContainer.appendChild(playIcon);
            } else {
                // Image preview
                const img = document.createElement("img");
                img.src = previewImage.url;
                img.style.cssText = "width: 100%; height: 100%; object-fit: cover;";
                img.onerror = () => {
                    imgContainer.innerHTML = '<div style="color: #444; font-size: 32px;">📷</div>';
                };
                imgContainer.appendChild(img);
            }
        } else {
            imgContainer.innerHTML = '<div style="color: #444; font-size: 32px;">📷</div>';
        }

        card.appendChild(imgContainer);

        // Info
        const info = document.createElement("div");
        info.style.cssText = "padding: 12px;";

        // Title - allow up to 3 lines
        const title = document.createElement("div");
        title.textContent = model.name || "Unknown";
        title.style.cssText = `
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            line-height: 1.3;
            display: -webkit-box;
            -webkit-line-clamp: 3;
            -webkit-box-orient: vertical;
            overflow: hidden;
            word-break: break-word;
        `;
        title.title = model.name;
        info.appendChild(title);

        // Meta row
        const meta = document.createElement("div");
        meta.style.cssText = "display: flex; gap: 8px; align-items: center; flex-wrap: wrap;";

        // Type badge
        const typeBadge = document.createElement("span");
        typeBadge.textContent = model.type || "LORA";
        typeBadge.style.cssText = `
            font-size: 10px;
            background: #2a3a5a;
            color: #8ab4f8;
            padding: 2px 6px;
            border-radius: 4px;
        `;
        meta.appendChild(typeBadge);

        // Downloaded badge - check if any version matches local hashes
        const firstVersion = model.modelVersions?.[0];
        if (this.isModelDownloaded(firstVersion)) {
            const downloadedBadge = document.createElement("span");
            downloadedBadge.textContent = "✓ Downloaded";
            downloadedBadge.style.cssText = `
                font-size: 10px;
                background: #2a4a2a;
                color: #8af888;
                padding: 2px 6px;
                border-radius: 4px;
            `;
            meta.appendChild(downloadedBadge);
        }

        // Update Available badge - for local items with updates
        if (model._hasUpdate) {
            const updateBadge = document.createElement("span");
            updateBadge.textContent = "⬆ Update";
            updateBadge.title = `New version: ${model._updateInfo?.latest_version_name || 'Unknown'}`;
            updateBadge.style.cssText = `
                font-size: 10px;
                background: #4a3a2a;
                color: #f8a;
                padding: 2px 6px;
                border-radius: 4px;
                cursor: pointer;
            `;
            // Click to view update details
            updateBadge.onclick = (e) => {
                e.stopPropagation();
                if (model._modelId) {
                    this.loadModelDetails(model._modelId);
                }
            };
            meta.appendChild(updateBadge);
        }

        // Base model badge
        const baseModel = model.modelVersions?.[0]?.baseModel;
        if (baseModel) {
            const baseBadge = document.createElement("span");
            baseBadge.textContent = baseModel;
            baseBadge.style.cssText = `
                font-size: 10px;
                background: #3a2a4a;
                color: #c8a8f8;
                padding: 2px 6px;
                border-radius: 4px;
            `;
            meta.appendChild(baseBadge);
        }

        info.appendChild(meta);

        // Stats row
        const stats = document.createElement("div");
        stats.style.cssText = "display: flex; gap: 12px; margin-top: 10px; color: #666; font-size: 11px;";

        const downloads = model.stats?.downloadCount || 0;
        stats.innerHTML = `
            <span>&#8595; ${this.formatNumber(downloads)}</span>
            ${model.stats?.rating ? `<span>&#9733; ${model.stats.rating.toFixed(1)}</span>` : ""}
        `;

        info.appendChild(stats);

        // Quick download buttons
        const downloadRow = document.createElement("div");
        downloadRow.style.cssText = `
            display: flex;
            gap: 6px;
            margin-top: 10px;
            flex-wrap: wrap;
        `;

        // Get first version's download info (reuse firstVersion from above)
        const downloadUrl = firstVersion?.downloadUrl;
        const filename = firstVersion?.files?.[0]?.name || `${model.name}.safetensors`;
        const modelType = model.type || "LORA";
        const versionBaseModel = firstVersion?.baseModel || "";
        const sha256 = firstVersion?.files?.[0]?.hashes?.SHA256;

        // For local/downloaded-only items, always show delete button
        if (model._isLocal) {
            const delBtn = document.createElement("button");
            delBtn.innerHTML = "🗑 Delete";
            delBtn.title = "Delete from disk";
            delBtn.style.cssText = `
                flex: 1;
                padding: 6px 10px;
                background: linear-gradient(135deg, #6a2a2a 0%, #4a1a1a 100%);
                border: none;
                border-radius: 4px;
                color: #faa;
                font-size: 11px;
                font-weight: 600;
                cursor: pointer;
            `;
            delBtn.onclick = async (e) => {
                e.stopPropagation();
                const deleted = await this.deleteLoraAndRefresh(sha256, model.name, card, model._localPath);
            };
            downloadRow.appendChild(delBtn);

            // Also show slot buttons if opened from LoRA node
            if (this.targetNode && (modelType === "LORA" || modelType === "LoCon" || modelType === "DoRA")) {
                for (let slot = 1; slot <= 3; slot++) {
                    const slotBtn = document.createElement("button");
                    slotBtn.textContent = `→${slot}`;
                    slotBtn.title = `Load to Slot ${slot}`;
                    slotBtn.style.cssText = `
                        flex: 1;
                        min-width: 40px;
                        padding: 6px 8px;
                        background: linear-gradient(135deg, #2a4a6a 0%, #1a3a5a 100%);
                        border: none;
                        border-radius: 4px;
                        color: #fff;
                        font-size: 11px;
                        font-weight: 600;
                        cursor: pointer;
                    `;
                    slotBtn.onclick = (e) => {
                        e.stopPropagation();
                        if (sha256) {
                            this.loadDownloadedToSlot(sha256, slot, model.name, null);
                        }
                    };
                    downloadRow.appendChild(slotBtn);
                }
            }
        } else if (this.targetNode && (modelType === "LORA" || modelType === "LoCon" || modelType === "DoRA")) {
            // Check if already downloaded
            const isDownloaded = this.isModelDownloaded(firstVersion);

            // Show slot buttons when opened from LoRA stack
            for (let slot = 1; slot <= 3; slot++) {
                const slotBtn = document.createElement("button");
                slotBtn.textContent = isDownloaded ? `→${slot}` : `↓${slot}`;
                slotBtn.title = isDownloaded ? `Load to Slot ${slot}` : `Download & Load to Slot ${slot}`;
                slotBtn.style.cssText = `
                    flex: 1;
                    min-width: 40px;
                    padding: 6px 8px;
                    background: linear-gradient(135deg, ${isDownloaded ? '#2a4a6a 0%, #1a3a5a 100%' : '#2a5a2a 0%, #1a4a1a 100%'});
                    border: none;
                    border-radius: 4px;
                    color: #fff;
                    font-size: 11px;
                    font-weight: 600;
                    cursor: pointer;
                `;
                slotBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (isDownloaded && sha256) {
                        // Try to load - will fall back to download if file was deleted
                        this.loadDownloadedToSlot(sha256, slot, model.name, downloadUrl ? {
                            url: downloadUrl,
                            filename: filename,
                            modelType: modelType,
                            baseModel: versionBaseModel
                        } : null);
                    } else if (downloadUrl) {
                        // Need to download first
                        this.downloadAndLoadToSlot(downloadUrl, filename, modelType, versionBaseModel, slot, model.name, sha256);
                    }
                };
                downloadRow.appendChild(slotBtn);
            }
        } else {
            // Show download button or delete button if already downloaded
            const isDownloaded = this.isModelDownloaded(firstVersion);

            if (isDownloaded && sha256) {
                // Show delete button
                const delBtn = document.createElement("button");
                delBtn.innerHTML = "🗑 Delete";
                delBtn.title = "Delete from disk";
                delBtn.style.cssText = `
                    flex: 1;
                    padding: 6px 10px;
                    background: linear-gradient(135deg, #6a2a2a 0%, #4a1a1a 100%);
                    border: none;
                    border-radius: 4px;
                    color: #faa;
                    font-size: 11px;
                    font-weight: 600;
                    cursor: pointer;
                `;
                delBtn.onclick = (e) => {
                    e.stopPropagation();
                    this.deleteLora(sha256, model.name);
                };
                downloadRow.appendChild(delBtn);
            } else {
                // Show download button
                const dlBtn = document.createElement("button");
                dlBtn.innerHTML = "↓ Download";
                dlBtn.title = "Download this model";
                dlBtn.style.cssText = `
                    flex: 1;
                    padding: 6px 10px;
                    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
                    border: none;
                    border-radius: 4px;
                    color: #fff;
                    font-size: 11px;
                    font-weight: 600;
                    cursor: pointer;
                `;
                dlBtn.onclick = (e) => {
                    e.stopPropagation();
                    if (downloadUrl) {
                        this.startQuickDownload(downloadUrl, filename, modelType, versionBaseModel, model.name, null, sha256);
                    } else {
                        // Need to load details to get download URL
                        this.loadModelDetails(model.id);
                    }
                };
                downloadRow.appendChild(dlBtn);
            }
        }

        info.appendChild(downloadRow);
        card.appendChild(info);

        return card;
    }

    async startQuickDownload(downloadUrl, filename, modelType, baseModel, modelName, loadToSlot = null, sha256 = null) {
        // Quick download directly from card (without needing full model details)
        try {
            const response = await api.fetchApi("/donut/civitai/download", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    downloadUrl: downloadUrl,
                    modelType: modelType,
                    baseModel: baseModel,
                    filename: filename,
                    sha256: sha256  // Pass hash to save after download
                })
            });

            if (response.ok) {
                const data = await response.json();
                this.activeDownloads[data.downloadId] = {
                    id: data.downloadId,
                    name: modelName,
                    filename: filename,
                    filepath: data.savePath,
                    modelType: modelType,
                    loadToSlot: loadToSlot,
                    targetNode: this.targetNode,
                    status: "downloading",
                    sha256: sha256
                };
                this.startDownloadPolling(data.downloadId);
                this.showDownloadsPanel();
                return data.downloadId;
            } else {
                const error = await response.json();
                // Handle duplicate file error
                if (error.error === "duplicate") {
                    this.showNotification(
                        "Already Downloaded",
                        `File exists: ${error.existingFile}`,
                        5000
                    );
                    if (sha256) this.addLocalHash(sha256);
                    return "duplicate";
                }
                console.error(`[CivitAI Browser] Download failed: ${error.error}`);
                return null;
            }
        } catch (error) {
            console.error("[CivitAI Browser] Download error:", error);
            return null;
        }
    }

    async downloadAndLoadToSlot(downloadUrl, filename, modelType, baseModel, slot, modelName, sha256 = null) {
        // Start the download with slot info so it auto-loads when complete
        await this.startQuickDownload(downloadUrl, filename, modelType, baseModel, modelName, slot, sha256);
    }

    renderDetailView() {
        const content = this.dialog?.querySelector("#donut-civitai-content");
        if (!content || !this.currentModel) return;

        content.innerHTML = "";
        const model = this.currentModel;

        // Back button
        const backBtn = document.createElement("button");
        backBtn.innerHTML = "&larr; Back to results";
        backBtn.style.cssText = `
            background: none;
            border: none;
            color: #6a9fd4;
            font-size: 14px;
            cursor: pointer;
            padding: 0;
            margin-bottom: 20px;
        `;
        backBtn.onclick = () => {
            this.currentView = "grid";
            this.renderGrid(true);  // Restore scroll position
        };
        content.appendChild(backBtn);

        // Main container
        const container = document.createElement("div");
        container.style.cssText = "display: flex; gap: 30px; flex-wrap: wrap;";

        // Left side - Images
        const leftSide = document.createElement("div");
        leftSide.style.cssText = "flex: 1; min-width: 300px; max-width: 500px;";

        // Main image
        const mainImageContainer = document.createElement("div");
        mainImageContainer.style.cssText = `
            width: 100%;
            aspect-ratio: 1;
            background: #0d0d1a;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        // Find the initial version to display (may be preselected from URL)
        let initialVersionIndex = 0;
        if (model._preselectedVersionId && model.modelVersions?.length) {
            const foundIndex = model.modelVersions.findIndex(v => v.id === model._preselectedVersionId);
            if (foundIndex >= 0) {
                initialVersionIndex = foundIndex;
                console.log(`[CivitAI Browser] Preselected version ${model._preselectedVersionId} found at index ${foundIndex}`);
            }
        }
        const version = model.modelVersions?.[initialVersionIndex];
        const mainImage = version?.images?.[0];
        if (mainImage?.url) {
            if (mainImage.type === "video") {
                const video = document.createElement("video");
                video.id = "donut-civitai-main-media";
                video.src = mainImage.url;
                video.style.cssText = "max-width: 100%; max-height: 100%; object-fit: contain;";
                video.controls = true;
                video.muted = true;
                video.loop = true;
                mainImageContainer.appendChild(video);
            } else {
                const img = document.createElement("img");
                img.id = "donut-civitai-main-media";
                img.src = mainImage.url;
                img.style.cssText = "max-width: 100%; max-height: 100%; object-fit: contain; cursor: zoom-in;";
                img.onclick = () => this.showLightbox(img.src);
                mainImageContainer.appendChild(img);
            }
        } else {
            mainImageContainer.innerHTML = '<div style="color: #444; font-size: 64px;">📷</div>';
        }

        leftSide.appendChild(mainImageContainer);

        // Thumbnail gallery
        const images = version?.images || [];
        if (images.length > 1) {
            const thumbnails = document.createElement("div");
            thumbnails.style.cssText = "display: flex; gap: 8px; margin-top: 10px; overflow-x: auto; padding: 5px 0;";

            for (let i = 0; i < Math.min(images.length, 8); i++) {
                const isVideo = images[i].type === "video";
                const thumb = document.createElement("div");
                thumb.style.cssText = `
                    width: 60px;
                    height: 60px;
                    min-width: 60px;
                    background: #0d0d1a;
                    border-radius: 6px;
                    overflow: hidden;
                    cursor: pointer;
                    border: 2px solid ${i === 0 ? '#6a9fd4' : 'transparent'};
                    position: relative;
                `;

                const thumbImg = document.createElement("img");
                thumbImg.src = images[i].url;
                thumbImg.style.cssText = "width: 100%; height: 100%; object-fit: cover;";

                // Video indicator
                if (isVideo) {
                    const videoIcon = document.createElement("div");
                    videoIcon.innerHTML = "▶";
                    videoIcon.style.cssText = "position: absolute; bottom: 2px; right: 2px; color: white; font-size: 10px; text-shadow: 0 1px 2px rgba(0,0,0,0.5);";
                    thumb.appendChild(videoIcon);
                }

                thumb.onclick = () => {
                    const mainMedia = document.getElementById("donut-civitai-main-media");
                    if (mainMedia) {
                        // Replace with correct element type
                        const newMedia = isVideo
                            ? this.createVideoElement(images[i].url)
                            : this.createImageElement(images[i].url);
                        mainMedia.replaceWith(newMedia);
                    }
                    thumbnails.querySelectorAll(":scope > div").forEach((t, idx) => {
                        t.style.borderColor = idx === i ? "#6a9fd4" : "transparent";
                    });
                };

                thumb.appendChild(thumbImg);
                thumbnails.appendChild(thumb);
            }

            leftSide.appendChild(thumbnails);
        }

        container.appendChild(leftSide);

        // Right side - Info
        const rightSide = document.createElement("div");
        rightSide.style.cssText = "flex: 1; min-width: 300px;";

        // Title
        const title = document.createElement("h2");
        title.textContent = model.name;
        title.style.cssText = "margin: 0 0 10px 0; font-size: 24px; font-weight: 600;";
        rightSide.appendChild(title);

        // Creator
        if (model.creator?.username) {
            const creator = document.createElement("div");
            creator.style.cssText = "font-size: 14px; color: #888; margin-bottom: 15px;";

            const byText = document.createTextNode("by ");
            creator.appendChild(byText);

            const creatorLink = document.createElement("span");
            creatorLink.textContent = model.creator.username;
            creatorLink.style.cssText = "color: #6a9fd4; cursor: pointer; text-decoration: underline;";
            creatorLink.title = `View all models by ${model.creator.username}`;
            creatorLink.onclick = () => {
                // Set creator filter and go back to grid
                this.filters.username = model.creator.username;
                this.filters.downloadedOnly = false;
                this.filters.updatesOnly = false;

                // Update the creator input in sidebar if it exists
                const creatorInput = this.dialog?.querySelector("#donut-civitai-creator-filter");
                if (creatorInput) {
                    creatorInput.value = model.creator.username;
                    // Show the clear button
                    const clearBtn = creatorInput.parentElement?.querySelector("span");
                    if (clearBtn) clearBtn.style.display = "block";
                }

                this.resetPagination();
                this.currentView = "grid";
                this.search();
            };
            creator.appendChild(creatorLink);

            rightSide.appendChild(creator);
        }

        // Badges
        const badges = document.createElement("div");
        badges.style.cssText = "display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px;";

        const typeBadge = document.createElement("span");
        typeBadge.textContent = model.type;
        typeBadge.style.cssText = "font-size: 12px; background: #2a3a5a; color: #8ab4f8; padding: 4px 10px; border-radius: 4px;";
        badges.appendChild(typeBadge);

        if (version?.baseModel) {
            const baseBadge = document.createElement("span");
            baseBadge.textContent = version.baseModel;
            baseBadge.style.cssText = "font-size: 12px; background: #3a2a4a; color: #c8a8f8; padding: 4px 10px; border-radius: 4px;";
            badges.appendChild(baseBadge);
        }

        // Downloaded badge (check if current version is downloaded)
        if (this.isModelDownloaded(version)) {
            const downloadedBadge = document.createElement("span");
            downloadedBadge.textContent = "✓ Downloaded";
            downloadedBadge.style.cssText = "font-size: 12px; background: #2a4a2a; color: #8af888; padding: 4px 10px; border-radius: 4px;";
            badges.appendChild(downloadedBadge);
        }

        rightSide.appendChild(badges);

        // Stats
        const stats = document.createElement("div");
        stats.style.cssText = "display: flex; gap: 20px; margin-bottom: 20px; color: #888; font-size: 13px;";
        stats.innerHTML = `
            <span>&#8595; ${this.formatNumber(model.stats?.downloadCount || 0)} downloads</span>
            ${model.stats?.rating ? `<span>&#9733; ${model.stats.rating.toFixed(1)} (${model.stats.ratingCount} ratings)</span>` : ""}
        `;
        rightSide.appendChild(stats);

        // Description
        if (model.description) {
            const desc = document.createElement("div");
            desc.style.cssText = "color: #aaa; font-size: 13px; line-height: 1.6; margin-bottom: 20px; max-height: 150px; overflow-y: auto;";
            // Strip HTML tags
            desc.textContent = model.description.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
            rightSide.appendChild(desc);
        }

        // Trigger words
        if (version?.trainedWords?.length > 0) {
            const triggerSection = document.createElement("div");
            triggerSection.style.cssText = "margin-bottom: 20px;";

            const triggerLabel = document.createElement("div");
            triggerLabel.textContent = "Trigger Words";
            triggerLabel.style.cssText = "font-size: 12px; font-weight: 600; color: #888; text-transform: uppercase; margin-bottom: 8px;";
            triggerSection.appendChild(triggerLabel);

            const triggerWords = document.createElement("div");
            triggerWords.style.cssText = `
                background: #0d0d1a;
                padding: 10px;
                border-radius: 6px;
                font-family: monospace;
                font-size: 12px;
                color: #8ab4f8;
            `;
            triggerWords.textContent = version.trainedWords.join(", ");
            triggerSection.appendChild(triggerWords);

            rightSide.appendChild(triggerSection);
        }

        // Version selector
        if (model.modelVersions?.length > 1) {
            const versionSection = document.createElement("div");
            versionSection.style.cssText = "margin-bottom: 20px;";

            const versionLabel = document.createElement("div");
            versionLabel.textContent = "Version";
            versionLabel.style.cssText = "font-size: 12px; font-weight: 600; color: #888; text-transform: uppercase; margin-bottom: 8px;";
            versionSection.appendChild(versionLabel);

            const versionSelect = document.createElement("select");
            versionSelect.id = "donut-civitai-version";
            versionSelect.style.cssText = `
                width: 100%;
                padding: 10px;
                background: #0d0d1a;
                border: 1px solid #333;
                border-radius: 6px;
                color: #eee;
                font-size: 14px;
                cursor: pointer;
            `;

            for (let i = 0; i < model.modelVersions.length; i++) {
                const v = model.modelVersions[i];
                const opt = document.createElement("option");
                opt.value = i;
                opt.textContent = `${v.name} - ${v.baseModel}`;
                if (i === initialVersionIndex) {
                    opt.selected = true;
                }
                versionSelect.appendChild(opt);
            }

            versionSection.appendChild(versionSelect);
            rightSide.appendChild(versionSection);
        }

        // File info & Download
        const file = version?.files?.[0];
        if (file) {
            const downloadSection = document.createElement("div");
            downloadSection.style.cssText = `
                background: #0d0d1a;
                padding: 15px;
                border-radius: 8px;
                margin-top: 20px;
            `;

            const fileInfo = document.createElement("div");
            fileInfo.style.cssText = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;";
            fileInfo.innerHTML = `
                <span style="font-size: 13px; color: #aaa;">${file.name}</span>
                <span style="font-size: 12px; color: #666;">${this.formatBytes(file.sizeKB * 1024)}</span>
            `;
            downloadSection.appendChild(fileInfo);

            // Check if this is a LoRA type and we have a target node
            const isLoraType = ["LORA", "LoCon", "DoRA"].includes(model.type);

            console.log(`[CivitAI Browser] Detail view - isLoraType: ${isLoraType}, targetNode: ${this.targetNode ? this.targetNode.type : 'null'}`);

            if (isLoraType && this.targetNode) {
                // Check if current version is downloaded
                const isDownloaded = this.isModelDownloaded(version);
                const sha256 = file?.hashes?.SHA256;

                // Show "Load to Slot" or "Download & Load to Slot" buttons
                const slotLabel = document.createElement("div");
                slotLabel.style.cssText = "font-size: 12px; color: #888; margin-bottom: 8px;";
                slotLabel.textContent = isDownloaded ? "Load to Slot:" : "Download & Load to Slot:";
                downloadSection.appendChild(slotLabel);

                const slotBtnContainer = document.createElement("div");
                slotBtnContainer.style.cssText = "display: flex; gap: 8px; margin-bottom: 10px;";

                for (let slot = 1; slot <= 3; slot++) {
                    const slotBtn = document.createElement("button");
                    slotBtn.innerHTML = isDownloaded ? `→ Slot ${slot}` : `↓ Slot ${slot}`;
                    const bgColor = isDownloaded ? "#4a6a8a" : "#5a8a5a";
                    const hoverColor = isDownloaded ? "#5a7a9a" : "#6a9a6a";
                    slotBtn.style.cssText = `
                        flex: 1;
                        padding: 10px;
                        background: ${bgColor};
                        border: none;
                        border-radius: 6px;
                        color: #fff;
                        font-size: 13px;
                        font-weight: 600;
                        cursor: pointer;
                        transition: background 0.2s;
                    `;
                    slotBtn.onmouseenter = () => !slotBtn.disabled && (slotBtn.style.background = hoverColor);
                    slotBtn.onmouseleave = () => !slotBtn.disabled && (slotBtn.style.background = bgColor);
                    slotBtn.onclick = () => {
                        const versionSelect = document.getElementById("donut-civitai-version");
                        const versionIndex = versionSelect ? parseInt(versionSelect.value) : 0;
                        const selectedVersion = model.modelVersions[versionIndex];
                        const selectedFile = selectedVersion?.files?.[0];
                        const selectedSha256 = selectedFile?.hashes?.SHA256;

                        if (this.isModelDownloaded(selectedVersion) && selectedSha256) {
                            // Try to load - will fall back to download if file was deleted
                            this.loadDownloadedToSlot(selectedSha256, slot, model.name, {
                                url: selectedVersion.downloadUrl,
                                filename: selectedFile.name,
                                modelType: model.type,
                                baseModel: selectedVersion.baseModel
                            });
                        } else {
                            this.startDownload(selectedVersion, slot, slotBtn);
                        }
                    };
                    slotBtnContainer.appendChild(slotBtn);
                }

                downloadSection.appendChild(slotBtnContainer);

                // Show delete button for downloaded LoRAs
                if (isDownloaded && sha256) {
                    const deleteBtn = document.createElement("button");
                    deleteBtn.innerHTML = "🗑 Delete from disk";
                    deleteBtn.style.cssText = `
                        width: 100%;
                        padding: 8px;
                        margin-bottom: 10px;
                        background: #5a2a2a;
                        border: none;
                        border-radius: 6px;
                        color: #faa;
                        font-size: 12px;
                        font-weight: 500;
                        cursor: pointer;
                        transition: background 0.2s;
                    `;
                    deleteBtn.onmouseenter = () => deleteBtn.style.background = "#7a3a3a";
                    deleteBtn.onmouseleave = () => deleteBtn.style.background = "#5a2a2a";
                    deleteBtn.onclick = () => {
                        const versionSelect = document.getElementById("donut-civitai-version");
                        const versionIndex = versionSelect ? parseInt(versionSelect.value) : 0;
                        const selectedVersion = model.modelVersions[versionIndex];
                        const selectedSha256 = selectedVersion?.files?.[0]?.hashes?.SHA256;
                        if (selectedSha256) {
                            this.deleteLora(selectedSha256, model.name);
                        }
                    };
                    downloadSection.appendChild(deleteBtn);
                }

                // Also show regular download button
                const orLabel = document.createElement("div");
                orLabel.style.cssText = "font-size: 11px; color: #666; text-align: center; margin: 8px 0;";
                orLabel.textContent = "— or —";
                downloadSection.appendChild(orLabel);
            }

            // API key warning if not configured
            if (!this.hasApiKey) {
                const apiWarning = document.createElement("div");
                apiWarning.style.cssText = `
                    background: #3a2a1a;
                    border: 1px solid #6a4a2a;
                    border-radius: 6px;
                    padding: 10px;
                    margin-bottom: 12px;
                    font-size: 12px;
                    color: #daa;
                `;
                apiWarning.innerHTML = `
                    <div style="font-weight: 600; margin-bottom: 4px;">⚠ CivitAI API Key Required</div>
                    <div style="color: #aaa;">Downloads may fail without an API key.</div>
                    <a href="https://civitai.com/user/account" target="_blank"
                       style="color: #6a9fd4; text-decoration: none; display: inline-block; margin-top: 6px;">
                       Get API key from CivitAI →
                    </a>
                    <div style="color: #888; font-size: 11px; margin-top: 4px;">
                        Then add it in ComfyUI Settings → DonutNodes
                    </div>
                `;
                downloadSection.appendChild(apiWarning);
            }

            const downloadBtn = document.createElement("button");
            downloadBtn.innerHTML = "&#8595; Download Only";
            downloadBtn.style.cssText = `
                width: 100%;
                padding: 12px;
                background: #4a6fa5;
                border: none;
                border-radius: 6px;
                color: #fff;
                font-size: 14px;
                font-weight: 600;
                cursor: pointer;
                transition: background 0.2s;
            `;
            downloadBtn.onmouseenter = () => !downloadBtn.disabled && (downloadBtn.style.background = "#5a7fb5");
            downloadBtn.onmouseleave = () => !downloadBtn.disabled && (downloadBtn.style.background = downloadBtn.dataset.originalBg || "#4a6fa5");
            downloadBtn.onclick = () => {
                const versionSelect = document.getElementById("donut-civitai-version");
                const versionIndex = versionSelect ? parseInt(versionSelect.value) : 0;
                const selectedVersion = model.modelVersions[versionIndex];
                this.startDownload(selectedVersion, null, downloadBtn);
            };

            downloadSection.appendChild(downloadBtn);

            // Show where it will be saved
            const savePath = document.createElement("div");
            savePath.style.cssText = "font-size: 11px; color: #666; margin-top: 10px; word-break: break-all;";
            savePath.textContent = `Will save to: ${this.getExpectedPath(model.type, version?.baseModel, file.name)}`;
            downloadSection.appendChild(savePath);

            rightSide.appendChild(downloadSection);
        }

        // CivitAI link
        const civitaiLink = document.createElement("a");
        civitaiLink.href = `https://civitai.com/models/${model.id}`;
        civitaiLink.target = "_blank";
        civitaiLink.style.cssText = "display: inline-block; margin-top: 15px; color: #6a9fd4; font-size: 13px; text-decoration: none;";
        civitaiLink.innerHTML = "View on CivitAI &rarr;";
        rightSide.appendChild(civitaiLink);

        container.appendChild(rightSide);
        content.appendChild(container);
    }

    getExpectedPath(modelType, baseModel, filename) {
        // This is just for display - actual path is computed server-side using folder_paths
        const folderNames = {
            "LORA": "loras",
            "LoCon": "loras",
            "DoRA": "loras",
            "Checkpoint": "checkpoints",
            "TextualInversion": "embeddings",
            "Controlnet": "controlnet",
            "Upscaler": "upscale_models",
            "VAE": "vae"
        };

        const baseModelFolders = {
            // SD 1.x
            "SD 1.5": "sd15", "SD 1.4": "sd15", "SD 1.5 LCM": "sd15", "SD 1.5 Hyper": "sd15",
            // SD 2.x
            "SD 2.0": "sd2", "SD 2.0 768": "sd2", "SD 2.1": "sd2", "SD 2.1 768": "sd2",
            // SDXL
            "SDXL 1.0": "sdxl", "SDXL 0.9": "sdxl", "SDXL Turbo": "sdxl", "SDXL Lightning": "sdxl",
            "SDXL Hyper": "sdxl", "SDXL 1.0 LCM": "sdxl", "SDXL Distilled": "sdxl",
            // Pony / NoobAI
            "Pony": "pony", "NoobAI": "noobai",
            // SD3
            "SD 3": "sd3", "SD 3.5": "sd3", "SD 3.5 Large": "sd3", "SD 3.5 Medium": "sd3",
            // Flux
            "Flux.1 D": "flux", "Flux.1 S": "flux", "Flux.1 Schnell": "flux",
            // Lumina / ZIT
            "Lumina": "zit", "Lumina2": "zit", "ZImageTurbo": "zit",
            // Other
            "Illustrious": "illustrious", "Hunyuan 1": "hunyuan", "AuraFlow": "auraflow"
        };

        const folderName = folderNames[modelType] || "loras";

        // For LoRAs, organize by base model subfolder
        if (["LORA", "LoCon", "DoRA"].includes(modelType)) {
            const subfolder = baseModelFolders[baseModel] || "";
            if (subfolder) {
                return `models/${folderName}/${subfolder}/${filename}`;
            }
        }

        return `models/${folderName}/${filename}`;
    }

    createPagination() {
        const pagination = document.createElement("div");
        pagination.style.cssText = `
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 10px;
            margin-top: 30px;
            padding: 20px 0;
        `;

        const prevBtn = document.createElement("button");
        prevBtn.textContent = "← Previous";
        prevBtn.style.cssText = this.getButtonStyle();
        prevBtn.disabled = this.currentPage <= 1;
        prevBtn.onclick = () => {
            if (this.currentPage > 1) {
                this.currentPage--;
                // Pop the last cursor to go back
                this.cursors.pop();
                const prevCursor = this.cursors.length > 0 ? this.cursors[this.cursors.length - 1] : null;
                this.search(prevCursor);
            }
        };
        pagination.appendChild(prevBtn);

        const pageInfo = document.createElement("span");
        pageInfo.textContent = `Page ${this.currentPage}`;
        pageInfo.style.cssText = "color: #888; font-size: 14px; padding: 0 15px;";
        pagination.appendChild(pageInfo);

        const nextBtn = document.createElement("button");
        nextBtn.textContent = "Next →";
        nextBtn.style.cssText = this.getButtonStyle();
        nextBtn.disabled = !this.hasNextPage;
        nextBtn.onclick = () => {
            if (this.hasNextPage && this.nextCursor) {
                // Save current cursor before moving to next page
                this.cursors.push(this.nextCursor);
                this.currentPage++;
                this.search(this.nextCursor);
            }
        };
        pagination.appendChild(nextBtn);

        return pagination;
    }

    updateLoadingState(isLoadingMore = false) {
        const content = this.dialog?.querySelector("#donut-civitai-content");
        if (!content) return;

        // Remove existing loading elements
        const existingOverlay = content.querySelector(".donut-loading-overlay");
        if (existingOverlay) {
            existingOverlay.remove();
        }
        const existingIndicator = content.querySelector(".donut-load-more-indicator");
        if (existingIndicator) {
            existingIndicator.remove();
        }

        if (this.isLoading) {
            if (this.endlessScroll && this.searchResults.length > 0) {
                // For endless scroll with existing results, show loading indicator at bottom
                const indicator = document.createElement("div");
                indicator.className = "donut-load-more-indicator";
                indicator.style.cssText = `
                    text-align: center;
                    padding: 20px;
                    color: #888;
                    font-size: 14px;
                `;
                indicator.innerHTML = `
                    <span style="display: inline-block; animation: spin 1s linear infinite;">↻</span>
                    Loading more...
                `;
                content.appendChild(indicator);
            } else {
                // Full overlay for initial load or pagination mode
                const overlay = document.createElement("div");
                overlay.className = "donut-loading-overlay";
                overlay.style.cssText = `
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(26, 26, 46, 0.8);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 10;
                `;
                overlay.innerHTML = `
                    <div style="text-align: center;">
                        <div style="font-size: 32px; animation: spin 1s linear infinite;">↻</div>
                        <div style="margin-top: 10px; color: #888;">Loading...</div>
                    </div>
                `;
                // Make content position relative for overlay positioning
                content.style.position = "relative";
                content.appendChild(overlay);
            }
        } else if (this.endlessScroll && this.hasNextPage) {
            // Show "scroll for more" indicator when not loading
            this.updateLoadMoreIndicator();
        }
    }

    getButtonStyle() {
        return `
            background: #333;
            color: #eee;
            border: 1px solid #444;
            border-radius: 6px;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 13px;
            transition: background 0.2s;
        `;
    }

    formatNumber(num) {
        if (num >= 1000000) return (num / 1000000).toFixed(1) + "M";
        if (num >= 1000) return (num / 1000).toFixed(1) + "K";
        return num.toString();
    }

    createImageElement(url) {
        const img = document.createElement("img");
        img.id = "donut-civitai-main-media";
        img.src = url;
        img.style.cssText = "max-width: 100%; max-height: 100%; object-fit: contain; cursor: zoom-in;";
        img.onclick = () => this.showLightbox(img.src);
        return img;
    }

    createVideoElement(url) {
        const video = document.createElement("video");
        video.id = "donut-civitai-main-media";
        video.src = url;
        video.style.cssText = "max-width: 100%; max-height: 100%; object-fit: contain;";
        video.controls = true;
        video.muted = true;
        video.loop = true;
        video.autoplay = true;
        return video;
    }

    formatBytes(bytes) {
        if (bytes >= 1073741824) return (bytes / 1073741824).toFixed(2) + " GB";
        if (bytes >= 1048576) return (bytes / 1048576).toFixed(1) + " MB";
        if (bytes >= 1024) return (bytes / 1024).toFixed(1) + " KB";
        return bytes + " B";
    }

    async show(onSelect = null, targetNode = null) {
        this.onSelect = onSelect;
        this.targetNode = targetNode;

        console.log(`[CivitAI Browser] Opening browser, targetNode: ${targetNode ? targetNode.type : 'null'}`);

        // Load local hashes for "downloaded" status checking (force reload each time browser opens)
        await this.loadLocalHashes(true);

        // Reset pagination for fresh search
        this.resetPagination();

        // If opened from a LoRA node, temporarily override filters
        // (these won't be saved - they'll be restored on close)
        if (targetNode) {
            this.filters.types = ["LORA", "LoCon", "DoRA"];

            // Try to detect base model from block_preset widget
            const detectedBaseModels = this.detectBaseModelFromNode(targetNode);
            if (detectedBaseModels.length > 0) {
                this.filters.baseModels = detectedBaseModels;
            }
        }

        const dialog = this.createDialog();
        document.body.appendChild(dialog);
        dialog.focus();

        // Initial search (don't save node-specific filters)
        if (targetNode) {
            // Temporarily disable saving for this search
            const origSave = this.saveFilters.bind(this);
            this.saveFilters = () => {};
            await this.search();
            this.saveFilters = origSave;
        } else {
            await this.search();
        }
    }

    detectBaseModelFromNode(node) {
        // Try to detect base model from any of the node's block_preset widgets
        if (!node || !node.widgets) return [];

        // Check all three block_preset widgets
        for (let i = 1; i <= 3; i++) {
            const presetWidget = node.widgets.find(w => w.name === `block_preset_${i}`);
            if (presetWidget && presetWidget.value && presetWidget.value !== "None") {
                const preset = presetWidget.value;

                // Map preset prefixes to CivitAI base model filters
                // Note: Use exact CivitAI filter names (with spaces)
                if (preset.startsWith("SDXL")) {
                    return ["SDXL 1.0"];
                } else if (preset.startsWith("SD15")) {
                    return ["SD 1.5"];
                } else if (preset.startsWith("ZIT")) {
                    return ["Z Image Turbo"];
                } else if (preset.startsWith("FLUX")) {
                    return ["Flux.1 D"];
                }
            }
        }

        return [];
    }

    showLightbox(imageUrl) {
        // Create lightbox overlay
        const lightbox = document.createElement("div");
        lightbox.id = "donut-civitai-lightbox";
        lightbox.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.95);
            z-index: 100001;
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: zoom-out;
        `;

        // Image container for zoom/pan
        const container = document.createElement("div");
        container.style.cssText = `
            position: relative;
            max-width: 95vw;
            max-height: 95vh;
            overflow: hidden;
        `;

        const img = document.createElement("img");
        img.src = imageUrl;
        img.style.cssText = `
            max-width: 95vw;
            max-height: 95vh;
            object-fit: contain;
            transform-origin: center center;
            transition: transform 0.1s ease-out;
        `;

        let scale = 1;
        let translateX = 0;
        let translateY = 0;
        let isDragging = false;
        let startX = 0;
        let startY = 0;

        // Mouse wheel zoom
        container.onwheel = (e) => {
            e.preventDefault();
            const delta = e.deltaY > 0 ? 0.9 : 1.1;
            scale = Math.min(Math.max(0.5, scale * delta), 5);
            img.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
        };

        // Drag to pan
        img.onmousedown = (e) => {
            if (scale > 1) {
                isDragging = true;
                startX = e.clientX - translateX;
                startY = e.clientY - translateY;
                img.style.cursor = "grabbing";
                e.stopPropagation();
            }
        };

        document.addEventListener("mousemove", (e) => {
            if (isDragging) {
                translateX = e.clientX - startX;
                translateY = e.clientY - startY;
                img.style.transform = `scale(${scale}) translate(${translateX}px, ${translateY}px)`;
            }
        });

        document.addEventListener("mouseup", () => {
            isDragging = false;
            img.style.cursor = scale > 1 ? "grab" : "zoom-out";
        });

        img.onclick = (e) => {
            if (scale === 1) {
                e.stopPropagation();
            }
        };

        // Close button
        const closeBtn = document.createElement("div");
        closeBtn.innerHTML = "✕";
        closeBtn.style.cssText = `
            position: absolute;
            top: 20px;
            right: 20px;
            color: white;
            font-size: 30px;
            cursor: pointer;
            z-index: 100002;
            padding: 10px;
        `;
        closeBtn.onclick = () => lightbox.remove();

        // Hint text
        const hint = document.createElement("div");
        hint.textContent = "Scroll to zoom • Drag to pan • Click outside to close";
        hint.style.cssText = `
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            color: #888;
            font-size: 12px;
        `;

        container.appendChild(img);
        lightbox.appendChild(container);
        lightbox.appendChild(closeBtn);
        lightbox.appendChild(hint);

        // Close on overlay click
        lightbox.onclick = (e) => {
            if (e.target === lightbox) {
                lightbox.remove();
            }
        };

        // Close on Escape
        lightbox.onkeydown = (e) => {
            if (e.key === "Escape") {
                lightbox.remove();
            }
        };

        document.body.appendChild(lightbox);
        lightbox.tabIndex = 0;
        lightbox.focus();
    }

    close() {
        if (this.dialog) {
            this.dialog.remove();
            this.dialog = null;
        }

        // If opened from a node context, restore filters from localStorage
        // (since we may have overridden them with node-specific filters)
        if (this.targetNode) {
            this.filters = this.loadFilters();
        }

        // Reset target node
        this.targetNode = null;
    }

    /**
     * Show a dialog for importing CivitAI URLs directly into LoRA slots
     * @param {object} targetNode - The DonutLoRAStack node to load into
     */
    async showUrlImportDialog(targetNode) {
        this.targetNode = targetNode;

        // Ensure local hashes are loaded for checking already-downloaded models
        await this.loadLocalHashes();

        // Create overlay
        const overlay = document.createElement("div");
        overlay.id = "donut-url-import-overlay";
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.7);
            z-index: 10000;
            display: flex;
            justify-content: center;
            align-items: center;
        `;

        // Create dialog
        const dialog = document.createElement("div");
        dialog.style.cssText = `
            background: #1a1a2e;
            border-radius: 12px;
            padding: 24px;
            width: 500px;
            max-width: 90vw;
            max-height: 80vh;
            overflow: hidden;
            display: flex;
            flex-direction: column;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
        `;

        // Header
        const header = document.createElement("div");
        header.style.cssText = "display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;";
        header.innerHTML = `
            <h2 style="margin: 0; color: #eee; font-size: 18px;">Import from CivitAI URLs</h2>
        `;

        const closeBtn = document.createElement("button");
        closeBtn.innerHTML = "✕";
        closeBtn.style.cssText = `
            background: none;
            border: none;
            color: #888;
            font-size: 20px;
            cursor: pointer;
            padding: 4px 8px;
        `;
        closeBtn.onmouseenter = () => closeBtn.style.color = "#fff";
        closeBtn.onmouseleave = () => closeBtn.style.color = "#888";
        closeBtn.onclick = () => overlay.remove();
        header.appendChild(closeBtn);
        dialog.appendChild(header);

        // Instructions
        const instructions = document.createElement("div");
        instructions.style.cssText = "color: #888; font-size: 13px; margin-bottom: 16px; line-height: 1.5;";
        instructions.innerHTML = `
            Paste CivitAI URLs (one per line) to download and load LoRAs into slots.<br>
            <span style="color: #666;">Example: https://civitai.com/models/12345/model-name?modelVersionId=67890</span>
        `;
        dialog.appendChild(instructions);

        // Textarea for URLs
        const textarea = document.createElement("textarea");
        textarea.id = "donut-url-import-input";
        textarea.placeholder = "Paste CivitAI URLs here (one per line)...";
        textarea.style.cssText = `
            width: 100%;
            height: 150px;
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #eee;
            font-size: 13px;
            font-family: monospace;
            padding: 12px;
            resize: vertical;
            margin-bottom: 16px;
            box-sizing: border-box;
        `;
        dialog.appendChild(textarea);

        // Slot selector
        const slotSection = document.createElement("div");
        slotSection.style.cssText = "margin-bottom: 16px;";

        const slotLabel = document.createElement("div");
        slotLabel.textContent = "Starting slot:";
        slotLabel.style.cssText = "color: #888; font-size: 13px; margin-bottom: 8px;";
        slotSection.appendChild(slotLabel);

        const slotSelect = document.createElement("select");
        slotSelect.id = "donut-url-import-slot";
        slotSelect.style.cssText = `
            background: #0d0d1a;
            border: 1px solid #333;
            border-radius: 6px;
            color: #eee;
            padding: 8px 12px;
            font-size: 14px;
            cursor: pointer;
        `;
        for (let i = 1; i <= 3; i++) {
            const opt = document.createElement("option");
            opt.value = i;
            opt.textContent = `Slot ${i}`;
            slotSelect.appendChild(opt);
        }
        slotSection.appendChild(slotSelect);

        const slotNote = document.createElement("div");
        slotNote.style.cssText = "color: #666; font-size: 12px; margin-top: 6px;";
        slotNote.textContent = "LoRAs will be loaded into consecutive slots starting from this one.";
        slotSection.appendChild(slotNote);

        dialog.appendChild(slotSection);

        // Status area
        const statusArea = document.createElement("div");
        statusArea.id = "donut-url-import-status";
        statusArea.style.cssText = `
            background: #0d0d1a;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 16px;
            max-height: 120px;
            overflow-y: auto;
            display: none;
            font-size: 13px;
            line-height: 1.6;
        `;
        dialog.appendChild(statusArea);

        // Buttons
        const buttons = document.createElement("div");
        buttons.style.cssText = "display: flex; gap: 12px; justify-content: flex-end;";

        const cancelBtn = document.createElement("button");
        cancelBtn.textContent = "Cancel";
        cancelBtn.style.cssText = `
            background: #333;
            border: none;
            border-radius: 6px;
            color: #eee;
            padding: 10px 20px;
            font-size: 14px;
            cursor: pointer;
        `;
        cancelBtn.onmouseenter = () => cancelBtn.style.background = "#444";
        cancelBtn.onmouseleave = () => cancelBtn.style.background = "#333";
        cancelBtn.onclick = () => overlay.remove();
        buttons.appendChild(cancelBtn);

        const importBtn = document.createElement("button");
        importBtn.textContent = "Download & Load";
        importBtn.style.cssText = `
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            border: none;
            border-radius: 6px;
            color: white;
            padding: 10px 20px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
        `;
        importBtn.onclick = () => this.processUrlImport(textarea, slotSelect, statusArea, importBtn, overlay);
        buttons.appendChild(importBtn);

        dialog.appendChild(buttons);
        overlay.appendChild(dialog);
        document.body.appendChild(overlay);

        // Focus textarea
        textarea.focus();

        // Close on escape
        const escHandler = (e) => {
            if (e.key === "Escape") {
                overlay.remove();
                document.removeEventListener("keydown", escHandler);
            }
        };
        document.addEventListener("keydown", escHandler);

        // Close on overlay click
        overlay.onclick = (e) => {
            if (e.target === overlay) overlay.remove();
        };
    }

    /**
     * Process the URL import - parse URLs, fetch model info, download, and load
     */
    async processUrlImport(textarea, slotSelect, statusArea, importBtn, overlay) {
        const urls = textarea.value.trim().split("\n").filter(line => line.trim());
        if (urls.length === 0) {
            this.showNotification("No URLs", "Please paste at least one CivitAI URL", 3000);
            return;
        }

        // Parse all URLs first
        const parsedUrls = [];
        for (const url of urls) {
            const parsed = this.parseCivitaiUrl(url.trim());
            if (parsed) {
                parsedUrls.push({ url: url.trim(), ...parsed });
            }
        }

        if (parsedUrls.length === 0) {
            this.showNotification("Invalid URLs", "No valid CivitAI model URLs found", 3000);
            return;
        }

        // Show status area and disable import button
        statusArea.style.display = "block";
        statusArea.innerHTML = "";
        importBtn.disabled = true;
        importBtn.style.opacity = "0.6";
        importBtn.textContent = "Processing...";
        textarea.disabled = true;
        slotSelect.disabled = true;

        let currentSlot = parseInt(slotSelect.value, 10);

        const addStatus = (text, color = "#aaa") => {
            const line = document.createElement("div");
            line.style.color = color;
            line.textContent = text;
            statusArea.appendChild(line);
            statusArea.scrollTop = statusArea.scrollHeight;
        };

        addStatus(`Found ${parsedUrls.length} valid URL(s), starting download...`);

        for (let i = 0; i < parsedUrls.length; i++) {
            const { modelId, versionId } = parsedUrls[i];

            if (currentSlot > 3) {
                addStatus(`⚠ Slot limit reached, skipping remaining URLs`, "#ffaa00");
                break;
            }

            addStatus(`[${i + 1}/${parsedUrls.length}] Fetching model ${modelId}...`);

            try {
                // Fetch model details
                const response = await api.fetchApi(`/donut/civitai/model/${modelId}`);
                if (!response.ok) {
                    addStatus(`✗ Failed to fetch model ${modelId}`, "#ff6666");
                    continue;
                }

                const model = await response.json();
                const modelName = model.name || `Model ${modelId}`;

                // Find the right version
                let version = model.modelVersions?.[0];
                if (versionId && model.modelVersions?.length) {
                    const found = model.modelVersions.find(v => v.id === versionId);
                    if (found) version = found;
                }

                if (!version) {
                    addStatus(`✗ No version found for ${modelName}`, "#ff6666");
                    continue;
                }

                const file = version.files?.[0];
                if (!file) {
                    addStatus(`✗ No downloadable file for ${modelName}`, "#ff6666");
                    continue;
                }

                // Check if it's a LoRA type
                if (!["LORA", "LoCon", "DoRA"].includes(model.type)) {
                    addStatus(`⚠ ${modelName} is not a LoRA (${model.type}), skipping`, "#ffaa00");
                    continue;
                }

                // Check if already downloaded
                const sha256 = file.hashes?.SHA256;
                if (sha256 && this.localHashes.has(sha256)) {
                    addStatus(`✓ ${modelName} already downloaded, loading to slot ${currentSlot}`, "#66ff66");
                    // Load directly using hash lookup
                    await this.loadDownloadedToSlot(sha256, currentSlot, modelName, {
                        url: version.downloadUrl,
                        filename: file.name,
                        modelType: model.type,
                        baseModel: version.baseModel
                    });
                    currentSlot++;
                    continue;
                }

                // Download
                addStatus(`↓ Downloading ${modelName} (${this.formatBytes(file.sizeKB * 1024)})...`);

                const downloadResponse = await api.fetchApi("/donut/civitai/download", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        downloadUrl: file.downloadUrl,
                        modelType: model.type,
                        baseModel: version.baseModel,
                        filename: file.name,
                        sha256: sha256
                    })
                });

                if (!downloadResponse.ok) {
                    const errorData = await downloadResponse.json().catch(() => ({}));
                    // Handle duplicate detected by server
                    if (errorData.error === "duplicate") {
                        addStatus(`✓ ${modelName} already exists: ${errorData.existingFile}`, "#66ff66");
                        if (sha256) this.addLocalHash(sha256);
                        // Try to load the existing file to slot
                        await this.loadDownloadedToSlot(sha256, currentSlot, modelName);
                        currentSlot++;
                        continue;
                    }
                    addStatus(`✗ Download failed: ${errorData.error || errorData.message || "Unknown error"}`, "#ff6666");
                    continue;
                }

                const downloadData = await downloadResponse.json();
                const downloadId = downloadData.downloadId;

                // Poll for completion
                let completed = false;
                let pollAttempts = 0;
                const maxAttempts = 300; // 5 minutes max
                let lastProgressPct = 0;

                while (!completed && pollAttempts < maxAttempts) {
                    await new Promise(resolve => setTimeout(resolve, 1000));
                    pollAttempts++;

                    try {
                        const progressResponse = await api.fetchApi(`/donut/civitai/download/status/${downloadId}`);
                        if (progressResponse.ok) {
                            const progress = await progressResponse.json();

                            // Update progress display every 10%
                            const pct = Math.round(progress.progress || 0);
                            if (pct > lastProgressPct + 9) {
                                lastProgressPct = pct;
                                // Update the last status line with progress
                                const lastLine = statusArea.lastElementChild;
                                if (lastLine && lastLine.textContent.startsWith("↓")) {
                                    lastLine.textContent = `↓ Downloading ${modelName} (${pct}%)...`;
                                }
                            }

                            if (progress.status === "completed") {
                                completed = true;
                                addStatus(`✓ Downloaded ${modelName}, loading to slot ${currentSlot}`, "#66ff66");

                                // Add to local hashes (uppercase for consistency)
                                if (sha256) this.addLocalHash(sha256);

                                // Refresh folder cache so the file appears in dropdowns
                                await this.refreshFolderCache(model.type);

                                // Load to slot using the filepath from download status
                                if (progress.filepath && this.targetNode) {
                                    await this.loadToNodeSlot(this.targetNode, currentSlot, progress.filepath);
                                }
                                currentSlot++;
                            } else if (progress.status === "failed" || progress.status === "error") {
                                addStatus(`✗ Download failed: ${progress.error || "Unknown error"}`, "#ff6666");
                                completed = true;
                            }
                        } else {
                            // If progress endpoint returns error, download might have finished already
                            // Check if the file exists by trying to reload hashes
                            console.log(`[URL Import] Progress check returned ${progressResponse.status}`);
                        }
                    } catch (e) {
                        console.log(`[URL Import] Poll error:`, e);
                        // Ignore poll errors, continue waiting
                    }
                }

                if (!completed) {
                    addStatus(`✗ Download timed out for ${modelName}`, "#ff6666");
                }

            } catch (error) {
                addStatus(`✗ Error processing model ${modelId}: ${error.message}`, "#ff6666");
            }
        }

        addStatus("Done!", "#66ff66");

        // Re-enable controls
        importBtn.disabled = false;
        importBtn.style.opacity = "1";
        importBtn.textContent = "Done";
        importBtn.onclick = () => overlay.remove();
    }
}

// Global instance
const civitaiBrowser = new DonutCivitaiBrowser();
window.DonutCivitaiBrowser = civitaiBrowser;

// Add CSS animations
const style = document.createElement("style");
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOut {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100px); opacity: 0; }
    }
    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }
    /* Responsive card widths */
    .donut-civitai-card {
        width: calc(25% - 12px) !important;
    }
    @media (max-width: 1400px) {
        .donut-civitai-card { width: calc(33.333% - 11px) !important; }
    }
    @media (max-width: 1100px) {
        .donut-civitai-card { width: calc(50% - 8px) !important; }
    }
    @media (max-width: 800px) {
        .donut-civitai-card { width: 100% !important; }
    }
`;
document.head.appendChild(style);

// Register extension for menu integration
app.registerExtension({
    name: "donut.CivitaiBrowser",

    async setup() {
        // Check if button should be shown (from settings)
        const showButton = app.ui.settings.getSettingValue("DonutNodes.CivitAI.ShowBrowserButton", true);

        // Load saved position from localStorage
        const savedPos = localStorage.getItem("donut-civitai-btn-pos");
        let posX = 20, posY = 20; // Default: bottom-right (as offsets from edges)
        let posAnchor = "bottom-right";

        if (savedPos) {
            try {
                const pos = JSON.parse(savedPos);
                posX = pos.x;
                posY = pos.y;
                posAnchor = pos.anchor || "bottom-right";
            } catch (e) {}
        }

        // Add floating CivitAI button to the UI
        const button = document.createElement("button");
        button.id = "donut-civitai-btn";
        button.innerHTML = "🌐 CivitAI";
        button.title = "Browse & Download from CivitAI (drag to move)";
        button.style.cssText = `
            position: fixed;
            ${posAnchor.includes("bottom") ? "bottom" : "top"}: ${posY}px;
            ${posAnchor.includes("right") ? "right" : "left"}: ${posX}px;
            z-index: 9999;
            background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 600;
            cursor: grab;
            box-shadow: 0 4px 15px rgba(37, 99, 235, 0.4);
            transition: box-shadow 0.2s;
            user-select: none;
            display: ${showButton ? "block" : "none"};
        `;

        // Dragging state
        let isDragging = false;
        let hasDragged = false;
        let startX, startY;
        let startLeft, startTop;

        button.onmousedown = (e) => {
            isDragging = true;
            hasDragged = false;
            startX = e.clientX;
            startY = e.clientY;

            // Get current position
            const rect = button.getBoundingClientRect();
            startLeft = rect.left;
            startTop = rect.top;

            // Switch to top-left positioning for dragging
            button.style.right = "auto";
            button.style.bottom = "auto";
            button.style.left = startLeft + "px";
            button.style.top = startTop + "px";

            button.style.cursor = "grabbing";
            button.style.boxShadow = "0 8px 25px rgba(37, 99, 235, 0.6)";
            e.preventDefault();
        };

        document.addEventListener("mousemove", (e) => {
            if (!isDragging) return;

            const dx = e.clientX - startX;
            const dy = e.clientY - startY;

            if (Math.abs(dx) > 3 || Math.abs(dy) > 3) {
                hasDragged = true;
            }

            let newLeft = startLeft + dx;
            let newTop = startTop + dy;

            // Keep within viewport
            const rect = button.getBoundingClientRect();
            newLeft = Math.max(0, Math.min(window.innerWidth - rect.width, newLeft));
            newTop = Math.max(0, Math.min(window.innerHeight - rect.height, newTop));

            button.style.left = newLeft + "px";
            button.style.top = newTop + "px";
        });

        document.addEventListener("mouseup", () => {
            if (!isDragging) return;
            isDragging = false;
            button.style.cursor = "grab";
            button.style.boxShadow = "0 4px 15px rgba(37, 99, 235, 0.4)";

            // Save position - determine which corner is closest
            const rect = button.getBoundingClientRect();
            const centerX = rect.left + rect.width / 2;
            const centerY = rect.top + rect.height / 2;
            const isRight = centerX > window.innerWidth / 2;
            const isBottom = centerY > window.innerHeight / 2;

            // Convert to edge-relative positioning
            let anchor, x, y;
            if (isRight && isBottom) {
                anchor = "bottom-right";
                x = window.innerWidth - rect.right;
                y = window.innerHeight - rect.bottom;
            } else if (!isRight && isBottom) {
                anchor = "bottom-left";
                x = rect.left;
                y = window.innerHeight - rect.bottom;
            } else if (isRight && !isBottom) {
                anchor = "top-right";
                x = window.innerWidth - rect.right;
                y = rect.top;
            } else {
                anchor = "top-left";
                x = rect.left;
                y = rect.top;
            }

            // Apply edge-relative positioning
            button.style.left = "auto";
            button.style.top = "auto";
            button.style.right = "auto";
            button.style.bottom = "auto";
            button.style[anchor.includes("bottom") ? "bottom" : "top"] = y + "px";
            button.style[anchor.includes("right") ? "right" : "left"] = x + "px";

            // Save to localStorage
            localStorage.setItem("donut-civitai-btn-pos", JSON.stringify({ x, y, anchor }));
        });

        button.onclick = (e) => {
            // Only open browser if we didn't drag
            if (!hasDragged) {
                civitaiBrowser.show();
            }
        };

        document.body.appendChild(button);
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Add to DonutLoRAStack right-click menu
        if (nodeData.name === "DonutLoRAStack") {
            const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions;
            nodeType.prototype.getExtraMenuOptions = function(_, options) {
                if (origGetExtraMenuOptions) {
                    origGetExtraMenuOptions.apply(this, arguments);
                }

                // Add separator
                options.push(null);

                // Capture 'this' (the node) for the callback
                const node = this;

                // Add CivitAI browser option
                options.push({
                    content: "Browse CivitAI",
                    callback: () => {
                        // Pass the node as the target so we get "Download & Load to Slot" buttons
                        civitaiBrowser.show((selectedModel) => {
                            console.log("[CivitAI] Selected model:", selectedModel);
                        }, node);
                    }
                });

                // Add URL import option
                options.push({
                    content: "Import from CivitAI URLs",
                    callback: () => {
                        civitaiBrowser.showUrlImportDialog(node);
                    }
                });
            };
        }
    }
});

export { DonutCivitaiBrowser };
