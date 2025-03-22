import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from itertools import product
import math

# Set page config
st.set_page_config(
    page_title="U-Value Calculator",
    page_icon="🧱",
    layout="wide",
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: rgb(0,48,60);
        text-align: center;
        margin-bottom: 1rem;
    }
    .subheader {
        font-size: 1.8rem;
        color: rgb(0,48,60);
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: rgb(207,241,242);
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 1rem;
    }
    .u-value-display {
        font-size: 2rem;
        font-weight: bold;
        color: rgb(211,69,29);
        text-align: center;
        margin: 1rem 0;
    }
    .layer-row {
        padding: 8px 0;
        border-radius: 5px;
    }
    .layer-row-alt {
        background-color: rgb(223,224,225);
    }
    .surface-resistance {
        background-color: rgb(136,219,223);
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        border-left: 4px solid rgb(0,163,173);
    }
    .best-config {
        background-color: rgb(207,241,242);
        padding: 15px;
        border-radius: 5px;
        margin: 15px 0;
        border-left: 4px solid rgb(0,163,173);
    }
    .target-header {
        background-color: rgb(207,241,242);
        padding: 15px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 20px;
        border: 2px solid rgb(0,163,173);
    }
    </style>
""", unsafe_allow_html=True)

# Constants
RSI_INT = 0.13  # Internal surface resistance (m²K/W)
RSE_EXT = 0.04  # External surface resistance (m²K/W)

# TT Colors
TT_Orange = "rgb(211,69,29)"
TT_Olive = "rgb(139,144,100)"
TT_LightBlue = "rgb(136,219,223)"
TT_MidBlue = "rgb(0,163,173)"
TT_DarkBlue = "rgb(0,48,60)"
TT_Grey = "rgb(99,102,105)"
TT_LightLightBlue = "rgb(207,241,242)"
TT_LightGrey = "rgb(223,224,225)"

# Material categories and colors
material_categories = {
    "Air": ["Air cavity", "DGU cavity 8-16-8"],
    "Insulation": ["Insulation - Mineral Wool", "Insulation - PF", "Insulation - PIR", "Insulation - EPS", "Insulation - XPS"],
    "Concrete": ["Concrete block", "Concrete reinforced (1%)", "Concrete reinforced (2%)", "Concrete high density", "Concrete medium density"],
    "Brick and Block": ["Blockwork - Fairfaced", "Brickwork ext. leaf", "Brickwork int. leaf"],
    "Timber": ["Timber low density", "Timber medium density", "Timber high density", "CLT", "MDF", "OSB", "Particleboard", 
               "Plywood low density", "Plywood medium density", "Plywood high density"],
    "Gypsum": ["Plaster", "Plasterboard medium density", "Plasterboard high density", "Plasterboard w/ vapour ret"],
    "Metal": ["Aluminium", "Aluminium foil 0.05mm", "Brass", "Bronze", "Copper", "Steel", "Stainless steel"],
    "Membrane": ["Breather membrane", "EPDM 0.75mm", "Bitumen", "Polyethilene"],
    "Stone and Ceramic": ["Stone", "Ceramic tiles", "Render", "Sand"],
    "Other": ["Glass", "Cement particleboard", "Polypropylene (Thermal pad)"]
}

# Material colors (realistic for visualization)
material_colors = {
    "Air cavity": "#E0F7FA",
    "DGU cavity 8-16-8": "#E1F5FE",
    "Aluminium": "#B0BEC5",
    "Aluminium foil 0.05mm": "#CFD8DC",
    "Bitumen": "#263238",
    "Blockwork - Fairfaced": "#BCAAA4",
    "Brickwork ext. leaf": "#D84315",
    "Brickwork int. leaf": "#E64A19",
    "Breather membrane": "#B3E5FC",
    "Brass": "#FFD54F",
    "Bronze": "#BF360C",
    "Cement particleboard": "#90A4AE",
    "Ceramic tiles": "#FF8A65",
    "CLT": "#8D6E63",
    "Concrete block": "#78909C",
    "Concrete reinforced (1%)": "#607D8B",
    "Concrete reinforced (2%)": "#546E7A",
    "Concrete high density": "#455A64",
    "Concrete medium density": "#78909C",
    "Copper": "#D84315",
    "EPDM 0.75mm": "#212121",
    "Glass": "#CFD8DC",
    "Insulation - Mineral Wool": "#FFECB3",
    "Insulation - PF": "#FFE082",
    "Insulation - PIR": "#FFD54F",
    "Insulation - EPS": "#FFECB3",
    "Insulation - XPS": "#FFE082",
    "MDF": "#A1887F",
    "OSB": "#8D6E63",
    "Particleboard": "#A1887F",
    "Plaster": "#F5F5F5",
    "Plasterboard medium density": "#EEEEEE",
    "Plasterboard high density": "#E0E0E0",
    "Plasterboard w/ vapour ret": "#EEEEEE",
    "Plywood low density": "#8D6E63",
    "Plywood medium density": "#795548",
    "Plywood high density": "#6D4C41",
    "Polypropylene (Thermal pad)": "#90CAF9",
    "Polyethilene": "#BBDEFB",
    "Render": "#ECEFF1",
    "Sand": "#FFCC80",
    "Steel": "#78909C",
    "Stainless steel": "#90A4AE",
    "Stone": "#9E9E9E",
    "Timber low density": "#D7CCC8",
    "Timber medium density": "#BCAAA4",
    "Timber high density": "#A1887F"
}

# Create materials dataframe
def create_materials_df():
    materials_data = {
        "Material": [
            "Air cavity", "Aluminium", "Aluminium foil 0.05mm", "Bitumen", "Blockwork - Fairfaced",
            "Brickwork ext. leaf", "Brickwork int. leaf", "Breather membrane", "Brass", "Bronze",
            "Cement particleboard", "Ceramic tiles", "CLT", "Concrete block", "Concrete reinforced (1%)",
            "Concrete reinforced (2%)", "Concrete high density", "Concrete medium density", "Copper",
            "DGU cavity 8-16-8", "EPDM 0.75mm", "Glass", "Insulation - Mineral Wool", "Insulation - PF",
            "Insulation - PIR", "Insulation - EPS", "Insulation - XPS", "MDF", "OSB", "Particleboard",
            "Plaster", "Plasterboard medium density", "Plasterboard high density", "Plasterboard w/ vapour ret",
            "Plywood low density", "Plywood medium density", "Plywood high density", "Polypropylene (Thermal pad)",
            "Polyethilene", "Render", "Sand", "Steel", "Stainless steel", "Stone",
            "Timber low density", "Timber medium density", "Timber high density"
        ],
        "Lambda [W/(m·K)]": [
            0.278, 160.0, 160.0, 0.23, 0.11,
            0.77, 0.56, 0.17, 120.0, 65.0,
            0.23, 1.3, 0.11, 1.35, 2.3,
            2.5, 2.0, 1.15, 380.0,
            0.02, 0.22, 1.0, 0.035, 0.02,
            0.022, 0.04, 0.034, 0.14, 0.13, 0.14,
            0.57, 0.21, 0.25, 0.19,
            0.09, 0.13, 0.17, 0.117,
            0.33, 0.8, 2.0, 50.0, 17.0, 3.5,
            0.12, 0.13, 0.18
        ]
    }
    
    # Add color column
    materials_data["Color"] = [material_colors[mat] for mat in materials_data["Material"]]
    
    # Add category column
    category_list = []
    for material in materials_data["Material"]:
        for category, materials in material_categories.items():
            if material in materials:
                category_list.append(category)
                break
    
    materials_data["Category"] = category_list
    
    return pd.DataFrame(materials_data)

# Create dropdown options with category headings
def create_material_options():
    options = []
    for category, materials in material_categories.items():
        options.append({"label": f"--- {category} ---", "disabled": True})
        for material in materials:
            options.append({"label": material, "disabled": False})
    return options

# Function to calculate U-value
def calculate_u_value(layers):
    # Calculate total thermal resistance including surface resistances
    total_r_value = RSI_INT + RSE_EXT
    
    for layer in layers:
        material = layer["material"]
        thickness = layer["thickness"] / 1000  # convert mm to m
        lambda_value = materials_df[materials_df["Material"] == material]["Lambda [W/(m·K)]"].values[0]
        r_value = thickness / lambda_value
        total_r_value += r_value
    
    # Calculate U-value (W/m²K)
    u_value = 1 / total_r_value
    return u_value

# Function to visualize wall buildup
def visualize_wall_buildup(layers):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    current_pos = 0
    legend_labels = []
    legend_patches = []
    
    # Create wall buildup visualization from inside to outside
    for layer in layers:
        material = layer["material"]
        thickness = layer["thickness"]
        color = materials_df[materials_df["Material"] == material]["Color"].values[0]
        
        # Add this layer to the wall
        ax.add_patch(plt.Rectangle((current_pos, 0), thickness, 1, color=color))
        current_pos += thickness
        
        # Add to legend
        legend_labels.append(f"{material} ({thickness} mm)")
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, color=color))
    
    # Set axes limits and labels
    ax.set_xlim(0, max(current_pos, 300))  # Set minimum width to 300mm for visibility
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Thickness (mm)")
    
    # Add interior and exterior labels
    ax.text(-10, 0.5, "Interior", ha="right", va="center", fontsize=12, fontweight="bold")
    ax.text(current_pos + 10, 0.5, "Exterior", ha="left", va="center", fontsize=12, fontweight="bold")
    
    # Add legend
    ax.legend(legend_patches, legend_labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    plt.tight_layout()
    return fig

# Function to run parametric analysis
def run_parametric_analysis(base_layers, parametric_configs, target_u_value):
    # Generate all combinations of parametric values
    param_values = []
    param_names = []
    
    for config in parametric_configs:
        layer_idx = config["layer_idx"]
        min_val = config["min_val"]
        max_val = config["max_val"]
        step = config["step"]
        
        # Generate values for this parameter
        values = list(range(min_val, max_val + step, step))
        param_values.append(values)
        
        # Create parameter name (material + thickness)
        material = base_layers[layer_idx]["material"]
        param_names.append(f"{material} (mm)")
    
    # Generate all combinations
    combinations = list(product(*param_values))
    
    # Create a list to store results
    results = []
    
    # Calculate U-value for each combination
    for combo in combinations:
        # Create a copy of base layers
        current_layers = base_layers.copy()
        
        # Update thicknesses for parametric layers
        for i, thickness in enumerate(combo):
            layer_idx = parametric_configs[i]["layer_idx"]
            current_layers[layer_idx] = current_layers[layer_idx].copy()
            current_layers[layer_idx]["thickness"] = thickness
        
        # Calculate U-value
        u_value = calculate_u_value(current_layers)
        
        # Calculate total thickness
        total_thickness = sum(layer["thickness"] for layer in current_layers)
        
        # Store results
        result = list(combo) + [u_value, total_thickness, u_value <= target_u_value]
        results.append(result)
    
    # Create dataframe
    column_names = param_names + ["U-Value (W/m²K)", "Total Thickness (mm)", "Meets Target"]
    results_df = pd.DataFrame(results, columns=column_names)
    
    return results_df

# Function to create parallel coordinates plot
def create_parallel_plot(results_df, target_u_value):
    # Create a copy of the dataframe for plotting
    plot_df = results_df.copy()
    
    # Create the parallel coordinates plot with color based on meeting target
    # Using color_continuous_scale instead of color_discrete_map
    fig = px.parallel_coordinates(
        plot_df, 
        color="U-Value (W/m²K)",  # Color by U-value instead
        color_continuous_scale=[[0, TT_MidBlue], [1, TT_Orange]],
        range_color=[plot_df["U-Value (W/m²K)"].min(), 
                    max(plot_df["U-Value (W/m²K)"].max(), target_u_value * 1.2)],
        dimensions=[col for col in plot_df.columns if col not in ["Meets Target"]],
        labels={col: col for col in plot_df.columns if col not in ["Meets Target"]},
    )
    
    # Add a horizontal line at the target U-value
    u_value_dim_index = [i for i, dim in enumerate(fig.data[0].dimensions) if dim.label == "U-Value (W/m²K)"][0]
    
    fig.add_shape(
        type="line",
        x0=u_value_dim_index,
        y0=target_u_value,
        x1=u_value_dim_index+0.01,
        y1=target_u_value,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    # Add annotation for target value
    fig.add_annotation(
        x=u_value_dim_index,
        y=target_u_value,
        text=f"Target: {target_u_value:.3f}",
        showarrow=True,
        arrowhead=1,
        ax=50,
        ay=-30,
        font=dict(size=12, color="red"),
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        margin=dict(l=80, r=80, t=50, b=50),
    )
    
    return fig

# Function to move a layer up in the list
def move_layer_up(idx):
    if idx > 0:
        st.session_state.layers[idx], st.session_state.layers[idx-1] = st.session_state.layers[idx-1], st.session_state.layers[idx]

# Function to move a layer down in the list
def move_layer_down(idx):
    if idx < len(st.session_state.layers) - 1:
        st.session_state.layers[idx], st.session_state.layers[idx+1] = st.session_state.layers[idx+1], st.session_state.layers[idx]

# Function to find optimal configurations
def find_optimal_configs(results_df):
    # Filter to only configurations that meet the target
    meeting_target = results_df[results_df["Meets Target"] == True]
    
    if meeting_target.empty:
        return None, None
    
    # Find the configuration with the lowest U-value (best insulation)
    best_insulation = meeting_target.loc[meeting_target["U-Value (W/m²K)"].idxmin()]
    
    # Find the configuration with the minimum thickness that meets the target
    shallowest = meeting_target.loc[meeting_target["Total Thickness (mm)"].idxmin()]
    
    return best_insulation, shallowest

# Create materials dataframe
materials_df = create_materials_df()

# Main application
st.markdown("<h1 class='main-header'>Wall U-Value Calculator</h1>", unsafe_allow_html=True)

# Add U-Value target input at the top
if 'target_u_value' not in st.session_state:
    st.session_state.target_u_value = 0.15

# Create a prominent target U-value section at the top
st.markdown("""
    <div class='target-header'>
        <h2>Target U-Value Setting</h2>
        <p>Set your target U-value for wall thermal performance assessment</p>
    </div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.session_state.target_u_value = st.number_input(
        "Target U-Value (W/m²K)",
        min_value=0.01,
        max_value=5.0,
        value=st.session_state.target_u_value,
        step=0.01,
        format="%.2f"
    )

# Information box with explanations
with st.expander("ℹ️ About U-Values and How to Use This Tool", expanded=False):
    st.markdown("""
        ### What is a U-Value?
        
        The U-value (thermal transmittance) measures how effective a material is as an insulator. 
        It's the rate of heat transfer through a structure divided by the difference in temperature 
        across that structure. The **lower** the U-value, the **better** the insulation performance.
        
        ### Calculation Method
        
        The U-value is calculated using the formula:
        
        $$U = \\frac{1}{R_{si} + \\sum_{j=1}^{n} \\frac{d_j}{\\lambda_j} + R_{se}}$$
        
        Where:
        - $R_{si}$ = Internal surface resistance (0.13 m²K/W)
        - $R_{se}$ = External surface resistance (0.04 m²K/W)
        - $d_j$ = Thickness of material layer $j$ in meters
        - $\\lambda_j$ = Thermal conductivity of material layer $j$ in W/(m·K)
        
        ### How to Use This Tool
        
        1. Set your **target U-value** at the top
        2. **Add layers** to your wall construction from interior to exterior
        3. For each layer, select a **material** and specify its **thickness**
        4. View the calculated **U-value** and visual representation of your wall
        5. Use the **Parametric Analysis** section to test different combinations of material thicknesses
        6. The **parallel coordinates plot** shows how different thickness combinations affect the U-value
        
        ### Building Regulations
        
        Typical target U-values (W/m²K) for walls in modern buildings:
        - Passive house standard: 0.10 - 0.15
        - New build (2021): 0.16 - 0.18
        - Building regulations minimum: 0.18 - 0.30
    """)

# Display tabs for different sections
tab1, tab2 = st.tabs(["Wall Configuration", "Parametric Analysis"])

with tab1:
    # Sidebar for materials selection and layer configuration
    st.markdown("<h2 class='subheader'>Wall Construction</h2>", unsafe_allow_html=True)
    
    # Initialize session state for layers
    if 'layers' not in st.session_state:
        st.session_state.layers = []
    
    # Function to add new layer
    def add_layer():
        st.session_state.layers.append({
            "material": "Brickwork ext. leaf",
            "thickness": 102
        })
    
    # Function to remove layer
    def remove_layer(idx):
        if idx < len(st.session_state.layers):
            st.session_state.layers.pop(idx)
    
    # Display current layers and allow editing
    if not st.session_state.layers:
        st.info("Start by adding layers to your wall construction from interior to exterior.")
    
    # Show internal surface resistance
    st.markdown(f"""
        <div class='surface-resistance'>
            <strong>INTERIOR</strong> - Internal Surface Resistance (R<sub>si</sub>) = {RSI_INT} m²K/W
        </div>
    """, unsafe_allow_html=True)
    
    # Create columns for layer configuration
    col_headers = st.columns([0.5, 3, 2, 1, 1.5])
    with col_headers[0]:
        st.write("**#**")
    with col_headers[1]:
        st.write("**Material**")
    with col_headers[2]:
        st.write("**Thickness (mm)**")
    with col_headers[3]:
        st.write("**λ [W/(m·K)]**")
    with col_headers[4]:
        st.write("**Actions**")
    
    # Create material options with categories
    material_options = create_material_options()
    
    # Display existing layers (from interior to exterior)
    for i, layer in enumerate(st.session_state.layers):
        row_class = "layer-row layer-row-alt" if i % 2 else "layer-row"
        
        st.markdown(f"<div class='{row_class}'>", unsafe_allow_html=True)
        
        cols = st.columns([0.5, 3, 2, 1, 1.5])
        
        with cols[0]:
            st.write(f"{i+1}")
        
        with cols[1]:
            # Combined category and material dropdown
            all_materials = []
            for category, materials in material_categories.items():
                all_materials.extend(materials)
            
            current_index = all_materials.index(layer["material"]) if layer["material"] in all_materials else 0
            
            # Create formatted options for selectbox
            formatted_options = []
            for opt in material_options:
                if opt["disabled"]:
                    formatted_options.append(f"{opt['label']}")
                else:
                    formatted_options.append(f"    {opt['label']}")
            
            # Find the index of the current material in formatted options
            selected_index = next((i for i, opt in enumerate(formatted_options) if opt.strip() == layer["material"]), 0)
            
            # Material selection
            selected_option = st.selectbox(
                f"Material {i+1}",
                options=formatted_options,
                index=selected_index,
                key=f"mat_{i}",
                label_visibility="collapsed"
            )
            
            # Update the material in the layer if a valid selection is made
            if selected_option.strip() in all_materials:
                layer["material"] = selected_option.strip()
        
        with cols[2]:
            # Thickness input
            layer["thickness"] = st.number_input(
                f"Thickness {i+1}",
                min_value=1,
                max_value=1000,
                value=layer["thickness"],
                key=f"thick_{i}",
                label_visibility="collapsed"
            )
        
        with cols[3]:
            # Display lambda value
            lambda_value = materials_df[materials_df["Material"] == layer["material"]]["Lambda [W/(m·K)]"].values[0]
            st.write(f"{lambda_value:.3f}")
        
        with cols[4]:
            # Action buttons with clear icons for moving layers
            action_cols = st.columns([1, 1, 1])
            with action_cols[0]:
                st.button("⬆️", key=f"up_{i}", on_click=move_layer_up, args=(i,), help="Move layer up (interior direction)")
            with action_cols[1]:
                st.button("⬇️", key=f"down_{i}", on_click=move_layer_down, args=(i,), help="Move layer down (exterior direction)")
            with action_cols[2]:
                st.button("🗑️", key=f"remove_{i}", on_click=remove_layer, args=(i,), help="Remove layer")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Add layer button
    st.button("Add Layer", on_click=add_layer)
    
    # Show external surface resistance
    st.markdown(f"""
        <div class='surface-resistance'>
            <strong>EXTERIOR</strong> - External Surface Resistance (R<sub>se</sub>) = {RSE_EXT} m²K/W
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate and display U-value
    if st.session_state.layers:
        u_value = calculate_u_value(st.session_state.layers)
        
        # Display U-value with coloring based on performance and target
        if u_value <= st.session_state.target_u_value:
            color = TT_MidBlue
            performance = "Passes Target ✅"
        else:
            color = TT_Orange
            performance = "Fails Target ❌"
        
        # Create two columns for U-value display and wall visualization
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown(f"""
                <div class='info-box'>
                    <h3>U-Value Result</h3>
                    <div class='u-value-display' style='color:{color};'>
                        {u_value:.3f} W/m²K
                    </div>
                    <p style='text-align:center;'>Performance: <strong>{performance}</strong></p>
                    <p style='text-align:center;'>Target: <strong>{st.session_state.target_u_value:.3f} W/m²K</strong></p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display total wall thickness
            total_thickness = sum(layer["thickness"] for layer in st.session_state.layers)
            st.markdown(f"""
                <div style='text-align:center; margin-top:1rem;'>
                    <p>Total Wall Thickness: <strong>{total_thickness} mm</strong></p>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Display wall buildup visualization
            if st.session_state.layers:
                fig = visualize_wall_buildup(st.session_state.layers)
                st.pyplot(fig)

with tab2:
    st.markdown("<h2 class='subheader'>Parametric Analysis</h2>", unsafe_allow_html=True)
    
    # Ensure we have layers before proceeding
    if not st.session_state.layers:
        st.warning("Please add at least one layer in the 'Wall Configuration' tab first.")
    else:
        st.info(f"""
        Use this section to test how changing the thickness of different layers affects the U-value.
        Select which layers to vary, set the thickness ranges, and run the analysis.
        
        Target U-Value: **{st.session_state.target_u_value:.3f} W/m²K**
        """)
        
        # Initialize parametric configs if not already in session state
        if 'parametric_configs' not in st.session_state:
            st.session_state.parametric_configs = []
        
        # Function to add parametric configuration
        def add_param_config():
            if st.session_state.parametric_configs:
                # Find unused layer indices
                used_indices = [config["layer_idx"] for config in st.session_state.parametric_configs]
                available_indices = [i for i in range(len(st.session_state.layers)) if i not in used_indices]
                
                if available_indices:
                    st.session_state.parametric_configs.append({
                        "layer_idx": available_indices[0],
                        "min_val": 50,
                        "max_val": 300,
                        "step": 25
                    })
                else:
                    st.warning("All layers are already being varied.")
            else:
                # Add first config
                st.session_state.parametric_configs.append({
                    "layer_idx": 0,
                    "min_val": 50,
                    "max_val": 300,
                    "step": 25
                })
        
        # Function to remove parametric configuration
        def remove_param_config(idx):
            if idx < len(st.session_state.parametric_configs):
                st.session_state.parametric_configs.pop(idx)
        
        # Display current parametric configurations
        st.subheader("Layer Variations")
        
        if not st.session_state.parametric_configs:
            st.info("Add at least one layer to vary for parametric analysis.")
        
        # Create columns for parametric configuration
        for i, config in enumerate(st.session_state.parametric_configs):
            col1, col2, col3, col4, col5, col6 = st.columns([3, 2, 2, 2, 2, 1])
            
            with col1:
                # Layer selection
                layer_options = [f"{idx+1}: {layer['material']} ({layer['thickness']} mm)" 
                                for idx, layer in enumerate(st.session_state.layers)]
                
                selected_idx = st.selectbox(
                    f"Layer {i+1}",
                    options=range(len(st.session_state.layers)),
                    format_func=lambda idx: layer_options[idx],
                    index=min(config["layer_idx"], len(st.session_state.layers)-1),
                    key=f"param_layer_{i}"
                )
                
                # Update the layer index in the config
                config["layer_idx"] = selected_idx
            
            with col2:
                # Minimum thickness
                config["min_val"] = st.number_input(
                    f"Min Thickness (mm) {i+1}",
                    min_value=1,
                    max_value=1000,
                    value=config["min_val"],
                    key=f"param_min_{i}"
                )
            
            with col3:
                # Maximum thickness
                config["max_val"] = st.number_input(
                    f"Max Thickness (mm) {i+1}",
                    min_value=1,
                    max_value=1000,
                    value=config["max_val"],
                    key=f"param_max_{i}"
                )
            
            with col4:
                # Step size
                config["step"] = st.number_input(
                    f"Step (mm) {i+1}",
                    min_value=1,
                    max_value=100,
                    value=config["step"],
                    key=f"param_step_{i}"
                )
            
            with col5:
                # Display number of steps
                num_steps = 1 + (config["max_val"] - config["min_val"]) // config["step"]
                st.write(f"Steps: {num_steps}")
            
            with col6:
                # Remove button
                st.button("🗑️", key=f"remove_param_{i}", on_click=remove_param_config, args=(i,))
        
        # Add parameter button
        st.button("Add Parameter", on_click=add_param_config)
        
        # Run analysis button
        if st.session_state.parametric_configs:
            total_combinations = 1
            for config in st.session_state.parametric_configs:
                num_steps = 1 + (config["max_val"] - config["min_val"]) // config["step"]
                total_combinations *= num_steps
            
            st.info(f"This analysis will run {total_combinations} combinations.")
            
            if st.button("Run Parametric Analysis"):
                with st.spinner("Running analysis..."):
                    # Run the parametric analysis
                    results = run_parametric_analysis(
                        st.session_state.layers, 
                        st.session_state.parametric_configs,
                        st.session_state.target_u_value
                    )
                    
                    # Create parallel coordinates plot
                    fig = create_parallel_plot(results, st.session_state.target_u_value)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Find optimal configurations
                    best_insulation, shallowest = find_optimal_configs(results)
                    
                    # Display optimal configurations if available
                    if best_insulation is not None and shallowest is not None:
                        # Create columns for the two optimal configurations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            <div class='best-config'>
                                <h3>Best Insulation Performance</h3>
                                <p>Configuration with the lowest U-value that meets the target</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display configuration details
                            st.metric("U-Value (W/m²K)", f"{best_insulation['U-Value (W/m²K)']:.3f}")
                            st.metric("Total Thickness", f"{best_insulation['Total Thickness (mm)']:.1f} mm")
                            
                            # Display layer thicknesses
                            for param_name in results.columns:
                                if param_name not in ["U-Value (W/m²K)", "Total Thickness (mm)", "Meets Target"]:
                                    st.write(f"{param_name}: {best_insulation[param_name]} mm")
                        
                        with col2:
                            st.markdown("""
                            <div class='best-config'>
                                <h3>Shallowest Configuration</h3>
                                <p>Thinnest wall that meets the target U-value</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Display configuration details
                            st.metric("U-Value (W/m²K)", f"{shallowest['U-Value (W/m²K)']:.3f}")
                            st.metric("Total Thickness", f"{shallowest['Total Thickness (mm)']:.1f} mm")
                            
                            # Display layer thicknesses
                            for param_name in results.columns:
                                if param_name not in ["U-Value (W/m²K)", "Total Thickness (mm)", "Meets Target"]:
                                    st.write(f"{param_name}: {shallowest[param_name]} mm")
                    
                    else:
                        st.warning("No configurations meet the target U-value. Try adjusting the parameters or target.")
                    
                    # Show results dataframe with filters
                    st.subheader("All Results")
                    
                    # Add filter for showing only configurations that meet the target
                    show_only_passing = st.checkbox("Show only configurations that meet the target", value=False)
                    
                    # Apply filter if checked
                    filtered_results = results[results["Meets Target"]] if show_only_passing else results
                    
                    # Sort by U-value
                    sorted_results = filtered_results.sort_values("U-Value (W/m²K)")
                    
                    # Display the table
                    st.dataframe(
                        sorted_results.style.apply(
                            lambda x: ['background-color: ' + TT_LightLightBlue if x['Meets Target'] else 'background-color: ' + TT_LightGrey for i in range(len(x))], 
                            axis=1
                        )
                    )
        else:
            st.warning("Please add at least one parameter to vary before running the analysis.")

# Add footer with information
st.markdown("""
    <div style='margin-top: 50px; text-align: center; color: #666;'>
        <hr>
        <p>Wall U-Value Calculator | Developed for building energy analysis</p>
    </div>
""", unsafe_allow_html=True)
