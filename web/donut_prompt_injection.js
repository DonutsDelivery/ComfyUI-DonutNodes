/**
 * DonutPromptInjection - Dynamic hierarchical dropdown handler
 *
 * This extension makes the subcategory and style dropdowns update dynamically
 * based on the selected main_category in the DonutPromptInjection node.
 *
 * Hierarchy: main_category -> subcategory -> style
 */

import { app } from "../../scripts/app.js";

// Cache for style hierarchy (fetched once from server)
let styleHierarchy = null;

// Cache for location hierarchy (fetched once from server)
let locationHierarchy = null;

// Cache for user's previous selections
const userSelections = {
    mainCategory: {},  // main_category -> last selected subcategory
    subcategory: {},   // "main_category:subcategory" -> last selected style
    locationCategory: {}  // location_category -> last selected location
};

/**
 * Fetch style hierarchy from the server
 */
async function fetchStyleHierarchy() {
    if (styleHierarchy !== null) {
        return styleHierarchy;
    }

    try {
        const response = await fetch('/donut/styles/hierarchy');
        if (response.ok) {
            styleHierarchy = await response.json();
            return styleHierarchy;
        }
    } catch (error) {
        console.error('[DonutPromptInjection] Failed to fetch style hierarchy:', error);
    }
    return null;
}

/**
 * Fetch location hierarchy from the server
 */
async function fetchLocationHierarchy() {
    if (locationHierarchy !== null) {
        return locationHierarchy;
    }

    try {
        const response = await fetch('/donut/locations/hierarchy');
        if (response.ok) {
            locationHierarchy = await response.json();
            return locationHierarchy;
        }
    } catch (error) {
        console.error('[DonutPromptInjection] Failed to fetch location hierarchy:', error);
    }
    return null;
}

/**
 * Update the subcategory widget options based on selected main category
 */
function updateSubcategoryOptions(node, mainCatWidget, subcatWidget, styleWidget, newMainCat) {
    if (!styleHierarchy) return;

    // Handle "None" main category - still show subcategory options but they won't matter
    if (newMainCat === "None") {
        subcatWidget.options.values = ["None", "Random"];
        subcatWidget.value = "None";
        updateStyleOptions(node, mainCatWidget, subcatWidget, styleWidget, "None");
        return;
    }

    // Handle "Random" main category - show all subcategories from all main categories
    let subcategories;
    if (newMainCat === "Random") {
        // Collect all unique subcategories across all main categories
        const allSubcats = new Set();
        for (const mainCat of Object.keys(styleHierarchy)) {
            for (const subcat of Object.keys(styleHierarchy[mainCat])) {
                allSubcats.add(subcat);
            }
        }
        subcategories = Array.from(allSubcats).sort();
    } else if (styleHierarchy[newMainCat]) {
        subcategories = Object.keys(styleHierarchy[newMainCat]);
    } else {
        return;
    }

    // Store current selection
    const oldMainCat = mainCatWidget._lastMainCategory;
    if (oldMainCat && subcatWidget.value) {
        userSelections.mainCategory[oldMainCat] = subcatWidget.value;
    }

    // Update subcategory options - always include None and Random first
    const newSubcats = ["None", "Random", ...subcategories];
    subcatWidget.options.values = newSubcats;

    // Restore previous selection or reset
    if (userSelections.mainCategory[newMainCat] && newSubcats.includes(userSelections.mainCategory[newMainCat])) {
        subcatWidget.value = userSelections.mainCategory[newMainCat];
    } else if (!newSubcats.includes(subcatWidget.value)) {
        subcatWidget.value = "None";
    }

    mainCatWidget._lastMainCategory = newMainCat;

    // Trigger subcategory update to refresh styles
    updateStyleOptions(node, mainCatWidget, subcatWidget, styleWidget, subcatWidget.value);
}

/**
 * Update the style widget options based on selected subcategory
 */
function updateStyleOptions(node, mainCatWidget, subcatWidget, styleWidget, newSubcat) {
    if (!styleHierarchy) return;

    const mainCat = mainCatWidget.value;

    // Handle "None" subcategory
    if (newSubcat === "None" || mainCat === "None") {
        styleWidget.options.values = ["None", "Random"];
        styleWidget.value = "None";
        return;
    }

    let styles = [];

    if (newSubcat === "Random") {
        // For Random subcategory, show all styles from the main category
        // (or all styles if main category is also Random)
        if (mainCat === "Random") {
            // Collect all styles from all subcategories of all main categories
            const allStyles = new Set();
            for (const mc of Object.keys(styleHierarchy)) {
                for (const sc of Object.keys(styleHierarchy[mc])) {
                    for (const style of styleHierarchy[mc][sc]) {
                        if (style !== "None") allStyles.add(style);
                    }
                }
            }
            styles = Array.from(allStyles).sort();
        } else if (styleHierarchy[mainCat]) {
            // Collect all styles from all subcategories of this main category
            const allStyles = new Set();
            for (const sc of Object.keys(styleHierarchy[mainCat])) {
                for (const style of styleHierarchy[mainCat][sc]) {
                    if (style !== "None") allStyles.add(style);
                }
            }
            styles = Array.from(allStyles).sort();
        }
    } else {
        // Specific subcategory selected
        if (mainCat === "Random") {
            // Find the subcategory in any main category
            for (const mc of Object.keys(styleHierarchy)) {
                if (styleHierarchy[mc][newSubcat]) {
                    styles = styleHierarchy[mc][newSubcat].filter(s => s !== "None");
                    break;
                }
            }
        } else if (styleHierarchy[mainCat] && styleHierarchy[mainCat][newSubcat]) {
            styles = styleHierarchy[mainCat][newSubcat].filter(s => s !== "None");
        }
    }

    // Store current selection
    const oldSubcat = subcatWidget._lastSubcategory;
    const oldKey = `${mainCat}:${oldSubcat}`;
    if (oldSubcat && styleWidget.value) {
        userSelections.subcategory[oldKey] = styleWidget.value;
    }

    // Update style options - always include None and Random first
    const newStyles = ["None", "Random", ...styles];
    styleWidget.options.values = newStyles;

    // Restore previous selection or reset
    const newKey = `${mainCat}:${newSubcat}`;
    if (userSelections.subcategory[newKey] && newStyles.includes(userSelections.subcategory[newKey])) {
        styleWidget.value = userSelections.subcategory[newKey];
    } else if (!newStyles.includes(styleWidget.value)) {
        styleWidget.value = "None";
    }

    subcatWidget._lastSubcategory = newSubcat;

    // Trigger widget change callback if it exists
    if (styleWidget.callback) {
        styleWidget.callback(styleWidget.value);
    }
}

/**
 * Update the location widget options based on selected location category
 */
function updateLocationOptions(node, locCatWidget, locationWidget, newLocCat) {
    if (!locationHierarchy) return;

    // Handle "None" location category
    if (newLocCat === "None") {
        locationWidget.options.values = ["None", "Random"];
        locationWidget.value = "None";
        return;
    }

    let locations = [];

    if (newLocCat === "Random") {
        // Collect all locations from all categories
        const allLocs = new Set();
        for (const category of Object.keys(locationHierarchy)) {
            for (const loc of locationHierarchy[category]) {
                if (loc !== "None") allLocs.add(loc);
            }
        }
        locations = Array.from(allLocs).sort();
    } else if (locationHierarchy[newLocCat]) {
        locations = locationHierarchy[newLocCat].filter(l => l !== "None");
    } else {
        return;
    }

    // Store current selection
    const oldLocCat = locCatWidget._lastLocationCategory;
    if (oldLocCat && locationWidget.value) {
        userSelections.locationCategory[oldLocCat] = locationWidget.value;
    }

    // Update location options - always include None and Random first
    const newLocations = ["None", "Random", ...locations];
    locationWidget.options.values = newLocations;

    // Restore previous selection or reset
    if (userSelections.locationCategory[newLocCat] && newLocations.includes(userSelections.locationCategory[newLocCat])) {
        locationWidget.value = userSelections.locationCategory[newLocCat];
    } else if (!newLocations.includes(locationWidget.value)) {
        locationWidget.value = "None";
    }

    locCatWidget._lastLocationCategory = newLocCat;

    // Trigger widget change callback if it exists
    if (locationWidget.callback) {
        locationWidget.callback(locationWidget.value);
    }
}

/**
 * Initialize the dynamic dropdowns for a node
 */
async function initializeDynamicDropdowns(node) {
    // Find style hierarchy widgets
    const mainCatWidget = node.widgets?.find(w => w.name === 'main_category');
    const subcatWidget = node.widgets?.find(w => w.name === 'subcategory');
    const styleWidget = node.widgets?.find(w => w.name === 'style');

    // Find location widgets
    const locCatWidget = node.widgets?.find(w => w.name === 'location_category');
    const locationWidget = node.widgets?.find(w => w.name === 'location');

    // Initialize style hierarchy dropdowns
    if (mainCatWidget && subcatWidget && styleWidget) {
        const hierarchy = await fetchStyleHierarchy();
        if (hierarchy) {
            // Store original callbacks
            const originalMainCatCallback = mainCatWidget.callback;
            const originalSubcatCallback = subcatWidget.callback;

            // Set up main category change handler
            mainCatWidget.callback = function(value) {
                updateSubcategoryOptions(node, mainCatWidget, subcatWidget, styleWidget, value);

                if (typeof originalMainCatCallback === 'function') {
                    originalMainCatCallback.call(this, value);
                }
            };

            // Set up subcategory change handler
            subcatWidget.callback = function(value) {
                updateStyleOptions(node, mainCatWidget, subcatWidget, styleWidget, value);

                if (typeof originalSubcatCallback === 'function') {
                    originalSubcatCallback.call(this, value);
                }
            };

            // Initialize with current values
            mainCatWidget._lastMainCategory = mainCatWidget.value;
            subcatWidget._lastSubcategory = subcatWidget.value;
            updateSubcategoryOptions(node, mainCatWidget, subcatWidget, styleWidget, mainCatWidget.value);
        } else {
            console.warn('[DonutPromptInjection] Could not load style hierarchy');
        }
    }

    // Initialize location dropdowns
    if (locCatWidget && locationWidget) {
        const locHierarchy = await fetchLocationHierarchy();
        if (locHierarchy) {
            // Store original callback
            const originalLocCatCallback = locCatWidget.callback;

            // Set up location category change handler
            locCatWidget.callback = function(value) {
                updateLocationOptions(node, locCatWidget, locationWidget, value);

                if (typeof originalLocCatCallback === 'function') {
                    originalLocCatCallback.call(this, value);
                }
            };

            // Initialize with current values
            locCatWidget._lastLocationCategory = locCatWidget.value;
            updateLocationOptions(node, locCatWidget, locationWidget, locCatWidget.value);
        } else {
            console.warn('[DonutPromptInjection] Could not load location hierarchy');
        }
    }
}

// Register the extension
app.registerExtension({
    name: "DonutNodes.PromptInjection",

    async nodeCreated(node) {
        if (node.comfyClass === "DonutPromptInjection") {
            // Small delay to ensure widgets are fully initialized
            setTimeout(() => {
                initializeDynamicDropdowns(node);
            }, 100);
        }
    },

    async loadedGraphNode(node) {
        // Also handle nodes loaded from saved workflows
        if (node.comfyClass === "DonutPromptInjection") {
            setTimeout(() => {
                initializeDynamicDropdowns(node);
            }, 100);
        }
    }
});
