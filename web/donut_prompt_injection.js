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

// Cache for user's previous selections
const userSelections = {
    mainCategory: {},  // main_category -> last selected subcategory
    subcategory: {}    // "main_category:subcategory" -> last selected style
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
 * Update the subcategory widget options based on selected main category
 */
function updateSubcategoryOptions(node, mainCatWidget, subcatWidget, styleWidget, newMainCat) {
    if (!styleHierarchy) return;

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

    // Update subcategory options
    const newSubcats = ["Random", ...subcategories];
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
                        allStyles.add(style);
                    }
                }
            }
            styles = Array.from(allStyles).sort();
        } else if (styleHierarchy[mainCat]) {
            // Collect all styles from all subcategories of this main category
            const allStyles = new Set();
            for (const sc of Object.keys(styleHierarchy[mainCat])) {
                for (const style of styleHierarchy[mainCat][sc]) {
                    allStyles.add(style);
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
                    styles = styleHierarchy[mc][newSubcat];
                    break;
                }
            }
        } else if (styleHierarchy[mainCat] && styleHierarchy[mainCat][newSubcat]) {
            styles = styleHierarchy[mainCat][newSubcat];
        }
    }

    // Store current selection
    const oldSubcat = subcatWidget._lastSubcategory;
    const oldKey = `${mainCat}:${oldSubcat}`;
    if (oldSubcat && styleWidget.value) {
        userSelections.subcategory[oldKey] = styleWidget.value;
    }

    // Update style options
    const newStyles = ["Random", ...styles];
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
 * Initialize the dynamic dropdowns for a node
 */
async function initializeDynamicDropdowns(node) {
    // Find the widgets
    const mainCatWidget = node.widgets?.find(w => w.name === 'main_category');
    const subcatWidget = node.widgets?.find(w => w.name === 'subcategory');
    const styleWidget = node.widgets?.find(w => w.name === 'style');

    if (!mainCatWidget || !subcatWidget || !styleWidget) {
        return;
    }

    // Fetch hierarchy if not already cached
    const hierarchy = await fetchStyleHierarchy();
    if (!hierarchy) {
        console.warn('[DonutPromptInjection] Could not load style hierarchy');
        return;
    }

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
