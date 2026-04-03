# DESIGN.md

# Design System Document

## 1. Overview & Creative North Star: "The Intellectual Canvas"
This design system moves beyond the "standard dashboard" by treating data as high-end editorial content. The Creative North Star is **The Intellectual Canvas**—a philosophy where the interface recedes to allow scholarly insights to breathe. We achieve this by rejecting the "boxed-in" layout of traditional data tools. Instead of rigid grids defined by lines, we use **Intentional Asymmetry** and **Tonal Depth** to guide the eye. The aesthetic is "Academic Premium": a marriage of the precision required for data science and the sophisticated clarity of a modern scientific journal.

---

### 2. Colors: Tonal Architecture
The palette is rooted in a "No-Line" philosophy. Boundaries are created through environmental shifts in color, not 1px strokes.

* **The "No-Line" Rule:** Explicitly prohibit `1px solid` borders for sectioning. Contrast between `surface` (#f8f9fa) and `surface_container_low` (#f3f4f5) must define structural boundaries.
* **Surface Hierarchy & Nesting:** Treat the UI as layers of fine paper.
* **Level 0 (Base):** `surface` (#f8f9fa) for the main dashboard background.
* **Level 1 (Sections):** `surface_container_low` (#f3f4f5) for large organizational blocks.
* **Level 2 (Active Cards):** `surface_container_lowest` (#ffffff) to provide a "pop" of clean white for primary data insights.
* **The Glass & Gradient Rule:** For the dark sidebar, use `inverse_surface` (#2e3132) with a subtle linear gradient transitioning into `on_surface_variant` (#414754) at a 15-degree angle. This adds "soul" to the dark navigation, preventing it from feeling like a flat "dead zone."
* **Accentuation:** Use `primary_container` (#1a73e8) for high-signal action items. For data visualization highlights, utilize `tertiary` (#9e4300) to provide a warm, sophisticated contrast to the cool blues.

---

### 3. Typography: Editorial Precision
We utilize a dual-typeface system to distinguish between "Exploration" (Data) and "Declaration" (Insights).

* **Display & Headlines (Manrope):** Use Manrope for all `display-` and `headline-` scales. Its geometric nature provides an authoritative, modern academic feel. Use `headline-md` (1.75rem) for major chart titles to give them an editorial presence.
* **The Body & UI (Inter):** Inter is our workhorse for legibility. Its high x-height is perfect for data-heavy tables and labels.
* **The "Information Density" Scale:**
* **Title-SM (1rem):** Used for card headers to maintain a clean, professional hierarchy.
* **Label-MD (0.75rem):** Reserved for axis labels and metadata, utilizing `on_surface_variant` (#414754) to keep the UI secondary to the data points.

---

### 4. Elevation & Depth: Tonal Layering
Traditional drop shadows are often too "heavy" for a clean academic aesthetic. We use **Ambient Elevation**.

* **The Layering Principle:** To lift a card, place a `surface_container_lowest` (#ffffff) element onto a `surface_container` (#edeeef) background. The change in hex value provides all the separation necessary.
* **Ambient Shadows:** Where floating elements (like Modals or Tooltips) are required, use a shadow with a `24px` blur and `4%` opacity, tinted with `primary` (#005bbf). This mimics the way light refracts through a lens.
* **The "Ghost Border" Fallback:** If a data table requires internal separation, use `outline_variant` (#c1c6d6) at **15% opacity**. It should be felt, not seen.
* **Glassmorphism:** Navigation overlays or floating filter bars should use `surface` (#f8f9fa) at 80% opacity with a `12px` backdrop-blur. This keeps the user grounded in their data context even when interacting with controls.

---

### 5. Components: The Research Kit

* **Buttons:**
* **Primary:** Solid `primary` (#005bbf) with `on_primary` (#ffffff) text. Use `roundedness.md` (0.375rem).
* **Secondary:** No background. Use `primary` text with a `surface_container_high` (#e7e8e9) hover state.
* **Data Cards:** Forbid internal dividers. Separate the "Header," "Metric," and "Footer" using the Spacing Scale (e.g., `spacing.6` between header and chart).
* **Chips:** Use `secondary_container` (#b2c9fe) with `on_secondary_container` (#3d5481). These should have a `full` (9999px) roundedness to contrast against the architectural squareness of the dashboard.
* **Input Fields:** Use `surface_container_highest` (#e1e3e4) as a subtle background fill instead of a border. On focus, transition the background to `surface_container_lowest` (#ffffff) and add a 1px `primary` ghost-border.
* **Academic Data Charts:**
* Ensure a 1:1.618 (Golden Ratio) aspect ratio for primary visualizations.
* Use the `spacing.8` (1.75rem) value for padding between the chart area and the card edge to ensure maximum "breathability."

---

### 6. Do's and Don'ts

#### Do:
* **Do** use white space as a structural element. If a section feels crowded, increase the spacing from `spacing.5` to `spacing.8` rather than adding a line.
* **Do** use `tertiary_fixed` (#ffdbcb) for warning states or "low-confidence" data intervals—it feels more sophisticated than a standard "warning orange."
* **Do** ensure all typography uses `on_surface` (#191c1d) for maximum contrast against the white backgrounds.

#### Don't:
* **Don't** use pure black (#000000) for text. It creates "visual vibration" against the white background. Stick to `on_surface`.
* **Don't** use 100% opaque borders. They create "visual noise" that distracts from complex data visualizations.
* **Don't** use standard "Material Blue" shadows. Always tint shadows with a hint of the `primary` color to maintain the academic brand identity.
