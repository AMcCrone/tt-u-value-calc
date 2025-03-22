import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import math
import itertools

# Set page configuration
st.set_page_config(
    page_title="Wall U-Value Calculator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Title and Introduction
st.title("Wall U-Value Calculator")
st.markdown("""
<div class="info-box">
This application helps you calculate the U-value of a wall assembly by selecting different materials and their thicknesses.
The U-value is a measure of thermal transmittance and lower values indicate better thermal insulation.
</div>
""", unsafe_allow_html=True)

# Create a database of materials with their properties
@st.cache_data
def load_materials_database():
    materials = {
        "External Surfaces": {
            "External Surface (Normal)": {"r_value": 0.04, "color": "#a9a9a9"},
            "External Surface (Sheltered)": {"r_value": 0.08, "color": "#808080"},
            "External Surface (Exposed)": {"r_value": 0.02, "color": "#d3d3d3"},
        },
        "Internal Surfaces": {
            "Internal Surface (Normal)": {"r_value": 0.13, "color": "#e9e9e9"},
            "Internal Surface (Low Emissivity)": {"r_value": 0.18, "color": "#f0f0f0"},
        },
        "Brick": {
            "Common Brick": {"r_value": 0.12, "color": "#b35a1f"},
            "Face Brick": {"r_value": 0.15, "color": "#bc4a0b"},
            "Engineering Brick": {"r_value": 0.10, "color": "#8b2e00"},
            "Clay Brick": {"r_value": 0.13, "color": "#cd5c5c"},
        },
        "Concrete": {
            "Dense Concrete": {"r_value": 0.07, "color": "#808080"},
            "Medium Concrete": {"r_value": 0.09, "color": "#a9a9a9"},
            "Lightweight Concrete": {"r_value": 0.11, "color": "#b8b8b8"},
            "Aerated Concrete Block": {"r_value": 0.18, "color": "#c4c4c4"},
        },
        "Stone": {
            "Limestone": {"r_value": 0.08, "color": "#d6cfc7"},
            "Granite": {"r_value": 0.05, "color": "#7a7d81"},
            "Sandstone": {"r_value": 0.09, "color": "#d8b894"},
            "Marble": {"r_value": 0.04, "color": "#f2f2f2"},
        },
        "Insulation": {
            "Expanded Polystyrene (EPS)": {"r_value": 0.85, "color": "#fffdd0"},
            "Extruded Polystyrene (XPS)": {"r_value": 1.0, "color": "#c8b560"},
            "Polyurethane Foam": {"r_value": 1.39, "color": "#f5f5dc"},
            "Mineral Wool": {"r_value": 0.91, "color": "#ffdb58"},
            "Glass Wool": {"r_value": 0.95, "color": "#f0e68c"},
            "Cellulose Fiber": {"r_value": 0.80, "color": "#deb887"},
        },
        "Wood": {
            "Softwood": {"r_value": 0.30, "color": "#deb887"},
            "Hardwood": {"r_value": 0.17, "color": "#a0522d"},
            "Plywood": {"r_value": 0.25, "color": "#d2b48c"},
            "OSB": {"r_value": 0.23, "color": "#daa06d"},
        },
        "Gypsum": {
            "Gypsum Plaster": {"r_value": 0.16, "color": "#f5f5f5"},
            "Gypsum Board": {"r_value": 0.18, "color": "#f8f8ff"},
        },
        "Air Gaps": {
            "Unventilated Air Gap (10mm)": {"r_value": 0.15, "color": "#e6e6fa"},
            "Unventilated Air Gap (25mm)": {"r_value": 0.18, "color": "#e6e6fa"},
            "Unventilated Air Gap (50mm+)": {"r_value": 0.18, "color": "#e6e6fa"},
            "Slightly Ventilated Air Gap": {"r_value": 0.09, "color": "#d8d8eb"},
        },
        "Metals": {
            "Steel": {"r_value": 0.0001, "color": "#71797E"},
            "Aluminum": {"r_value": 0.00002, "color": "#848789"},
        }
    }
    
    # Convert to dataframe for easier handling
    rows = []
    for category, category_materials in materials.items():
        for name, props in category_materials.items():
            rows.append({
                "Category": category,
                "Material": name,
                "R-Value (m²K/W)": props["r_value"],
                "Color": props["color"]
            })
    
    df = pd.DataFrame(rows)
    return df, materials

materials_df, materials_dict = load_materials_database()

# Calculate U-value function
def calculate_u_value(layers):
    total_r_value = 0
    
    for layer in layers:
        material = layer["material"]
        thickness = layer["thickness"]
        category = layer["category"]
        
        # Skip calculation for surface resistances (already in m²K/W)
        if category in ["External Surfaces", "Internal Surfaces"]:
            total_r_value += materials_dict[category][material]["r_value"]
        else:
            # For other materials, multiply r-value by thickness in meters
            total_r_value += materials_dict[category][material]["r_value"] * (thickness / 1000)
    
    if total_r_value == 0:
        return float('inf')
    
    # U-value is the reciprocal of the total R-value
    u_value = 1 / total_r_value
    return u_value

# Function to plot the wall section
def plot_wall_section(layers):
    if not layers:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 6))
    current_x = 0
    layer_positions = []
    
    # Plot each layer
    for i, layer in enumerate(layers):
        material = layer["material"]
        thickness = layer["thickness"]
        category = layer["category"]
        
        # Skip visualization for surface resistances (they don't have physical thickness)
        if category in ["External Surfaces", "Internal Surfaces"]:
            continue
            
        color = materials_dict[category][material]["color"]
        
        # Create rectangle
        rect = patches.Rectangle(
            (current_x, 0), thickness, 100, 
            linewidth=1, edgecolor='black', facecolor=color, alpha=0.8
        )
        ax.add_patch(rect)
        
        # Store position for labels
        layer_positions.append((current_x + thickness/2, material, thickness))
        current_x += thickness
    
    # Add labels
    for x, material, thickness in layer_positions:
        ax.text(x, 105, f"{thickness}mm", ha='center', va='bottom', rotation=90, fontsize=8)
        ax.text(x, 50, material.split(" ")[0], ha='center', va='center', fontsize=8, rotation=90)
    
    # Set axis limits and labels
    ax.set_xlim(0, current_x)
    ax.set_ylim(0, 120)
    ax.set_xlabel('Thickness (mm)')
    ax.set_xticks(np.arange(0, current_x+1, 50))
    ax.set_yticks([])
    ax.set_title('Wall Section')
    
    # Add annotations for external and internal sides
    if layers and current_x > 0:
        ax.text(0, -10, 'External', fontsize=10, ha='left')
        ax.text(current_x, -10, 'Internal', fontsize=10, ha='right')
        
        # Add arrows indicating heat flow
        ax.arrow(current_x/2, -20, -current_x/4, 0, head_width=5, head_length=10, 
                 fc='red', ec='red', width=1)
        ax.text(current_x/2, -30, 'Heat Flow (Winter)', color='red', ha='center')
    
    plt.tight_layout()
    return fig

# Function to create parallel coordinates plot
def create_parallel_coordinates_plot(df):
    if df.empty:
        return None
        
    # Create parallel coordinates plot
    fig = px.parallel_coordinates(
        df, 
        color="U-Value (W/m²K)",
        color_continuous_scale=px.colors.sequential.Viridis_r,  # Reversed so lower U-values are better (green)
        labels={col: col for col in df.columns},
        title="U-Value Optimization"
    )
    
    # Update layout
    fig.update_layout(
        coloraxis_colorbar=dict(
            title="U-Value (W/m²K)",
            tickvals=[df["U-Value (W/m²K)"].min(), df["U-Value (W/m²K)"].max()],
            ticktext=["Better", "Worse"],
        ),
        height=600,
    )
    
    return fig

# Application layout
st.markdown("## Wall Configuration")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Wall Builder", "Sensitivity Analysis", "Information"])

with tab1:
    st.markdown("### Build your wall layer by layer")
    st.markdown("Add layers from external (left) to internal (right) side of the wall.")
    
    # Initialize session state
    if 'layers' not in st.session_state:
        st.session_state.layers = []
    
    # Display current layers and U-value
    if st.session_state.layers:
        u_value = calculate_u_value(st.session_state.layers)
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.metric(
                "U-Value", 
                f"{u_value:.3f} W/m²K",
                help="Lower values indicate better thermal insulation"
            )
        with col2:
            st.metric(
                "R-Value", 
                f"{(1/u_value):.3f} m²K/W",
                help="Higher values indicate better thermal insulation"
            )
        with col3:
            # Evaluate the U-value against building regulations
            evaluation = ""
            if u_value < 0.16:
                evaluation = "Excellent (Passive House)"
                color = "green"
            elif u_value < 0.25:
                evaluation = "Very Good"
                color = "lightgreen"
            elif u_value < 0.3:
                evaluation = "Good"
                color = "blue"
            elif u_value < 0.35:
                evaluation = "Acceptable"
                color = "orange"
            else:
                evaluation = "Poor"
                color = "red"
                
            st.markdown(f"""
            <div style="border-radius: 5px; padding: 10px; background-color: {color}; color: white; text-align: center;">
                <span style="font-weight: bold; font-size: 16px;">Rating: {evaluation}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Display wall section
        fig = plot_wall_section(st.session_state.layers)
        if fig:
            st.pyplot(fig)
    
    # Add a new layer
    with st.expander("Add New Layer", expanded=not st.session_state.layers):
        col1, col2 = st.columns(2)
        
        with col1:
            selected_category = st.selectbox(
                "Material Category",
                options=sorted(materials_dict.keys()),
                key="category_select"
            )
        
        with col2:
            available_materials = list(materials_dict[selected_category].keys())
            selected_material = st.selectbox(
                "Material",
                options=available_materials,
                key="material_select"
            )
        
        # Get R-value for selected material
        r_value = materials_dict[selected_category][selected_material]["r_value"]
        
        # For surface resistances, we don't need thickness
        if selected_category in ["External Surfaces", "Internal Surfaces"]:
            thickness = 0
            st.info(f"Surface resistance: {r_value} m²K/W (No physical thickness)")
        else:
            # For regular materials, ask for thickness
            thickness = st.slider(
                "Thickness (mm)",
                min_value=1,
                max_value=500,
                value=100,
                step=1,
                key="thickness_slider"
            )
            
            equiv_r_value = r_value * (thickness / 1000)
            st.info(f"R-Value: {r_value} W/mK × {thickness}mm = {equiv_r_value:.3f} m²K/W")
        
        # Add layer
        if st.button("Add Layer"):
            st.session_state.layers.append({
                "category": selected_category,
                "material": selected_material,
                "thickness": thickness
            })
            st.rerun()  # Updated from st.experimental_rerun()
    
    # Display and edit current layers
    if st.session_state.layers:
        st.markdown("### Current Wall Layers")
        
        for i, layer in enumerate(st.session_state.layers):
            col1, col2, col3, col4 = st.columns([3, 3, 2, 1])
            
            with col1:
                st.markdown(f"**{layer['material']}**")
            
            with col2:
                if layer['category'] not in ["External Surfaces", "Internal Surfaces"]:
                    st.markdown(f"{layer['thickness']} mm")
                else:
                    st.markdown("Surface resistance")
            
            with col3:
                if layer['category'] in ["External Surfaces", "Internal Surfaces"]:
                    r_value = materials_dict[layer['category']][layer['material']]["r_value"]
                    st.markdown(f"R: {r_value:.3f} m²K/W")
                else:
                    r_value = materials_dict[layer['category']][layer['material']]["r_value"]
                    equiv_r_value = r_value * (layer['thickness'] / 1000)
                    st.markdown(f"R: {equiv_r_value:.3f} m²K/W")
            
            with col4:
                if st.button("Delete", key=f"delete_{i}"):
                    st.session_state.layers.pop(i)
                    st.rerun()  # Updated from st.experimental_rerun()
        
        # Clear all layers
        if st.button("Clear All Layers"):
            st.session_state.layers = []
            st.rerun()  # Updated from st.experimental_rerun()

# Sensitivity Analysis Tab
with tab2:
    st.markdown("### Sensitivity Analysis")
    st.markdown("""
    This section allows you to test different combinations of material thicknesses
    to find the optimal configuration for your wall.
    """)
    
    if not st.session_state.layers:
        st.warning("Please add some layers to your wall first in the Wall Builder tab.")
    else:
        # Select layers to vary
        non_surface_layers = [
            (i, layer) for i, layer in enumerate(st.session_state.layers)
            if layer['category'] not in ["External Surfaces", "Internal Surfaces"]
        ]
        
        if not non_surface_layers:
            st.warning("Please add some physical layers (not just surface resistances) to perform sensitivity analysis.")
        else:
            # Let user select which layers to vary
            st.markdown("#### Select layers to vary:")
            
            selected_layer_indices = []
            for i, layer in non_surface_layers:
                if st.checkbox(f"{layer['material']} ({layer['thickness']}mm)", key=f"vary_{i}"):
                    selected_layer_indices.append(i)
            
            if not selected_layer_indices:
                st.info("Select at least one layer to vary its thickness.")
            else:
                # Configure thickness ranges for selected layers
                st.markdown("#### Configure thickness ranges:")
                
                ranges = {}
                for i in selected_layer_indices:
                    layer = st.session_state.layers[i]
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        min_val = st.number_input(
                            f"Min thickness (mm) for {layer['material']}",
                            min_value=1,
                            max_value=500,
                            value=max(1, layer['thickness'] - 50),
                            step=1,
                            key=f"min_{i}"
                        )
                    
                    with col2:
                        max_val = st.number_input(
                            f"Max thickness (mm) for {layer['material']}",
                            min_value=1,
                            max_value=500,
                            value=min(500, layer['thickness'] + 50),
                            step=1,
                            key=f"max_{i}"
                        )
                    
                    with col3:
                        step_val = st.number_input(
                            f"Step size (mm)",
                            min_value=1,
                            max_value=100,
                            value=10,
                            step=1,
                            key=f"step_{i}"
                        )
                    
                    ranges[i] = (min_val, max_val, step_val)
                
                # Calculate button
                if st.button("Run Analysis"):
                    # Create all combinations
                    thickness_ranges = []
                    for i, (min_val, max_val, step_val) in ranges.items():
                        thickness_range = list(range(min_val, max_val + 1, step_val))
                        thickness_ranges.append((i, thickness_range))
                    
                    # Generate all combinations
                    combinations = list(itertools.product(*[range_values for _, range_values in thickness_ranges]))
                    
                    # Prepare results data
                    results = []
                    for combo in combinations:
                        # Create a copy of the current layers
                        temp_layers = st.session_state.layers.copy()
                        
                        # Apply the thickness combination
                        for idx, thickness in zip([i for i, _ in thickness_ranges], combo):
                            temp_layers[idx]["thickness"] = thickness
                        
                        # Calculate U-value
                        u_value = calculate_u_value(temp_layers)
                        
                        # Create a row for the results
                        row = {}
                        for idx, thickness in zip([i for i, _ in thickness_ranges], combo):
                            material = temp_layers[idx]["material"]
                            key = f"{material} (mm)"
                            row[key] = thickness
                        
                        row["U-Value (W/m²K)"] = u_value
                        results.append(row)
                    
                    # Create DataFrame
                    results_df = pd.DataFrame(results)
                    
                    # Display the parallel coordinates plot
                    st.markdown("#### Results:")
                    
                    if len(results) > 1:  # Only show plot if we have more than one result
                        fig = create_parallel_coordinates_plot(results_df)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Show the best configuration
                        best_config = results_df.loc[results_df["U-Value (W/m²K)"].idxmin()]
                        
                        st.markdown("#### Best Configuration:")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Best U-Value", f"{best_config['U-Value (W/m²K)']:.3f} W/m²K")
                        
                        with col2:
                            improvement = ((calculate_u_value(st.session_state.layers) - best_config['U-Value (W/m²K)']) / 
                                         calculate_u_value(st.session_state.layers)) * 100
                            st.metric("Improvement", f"{improvement:.1f}%")
                        
                        # Display best configuration details
                        st.markdown("##### Layer Thicknesses:")
                        for col in results_df.columns:
                            if col != "U-Value (W/m²K)":
                                st.markdown(f"- **{col}**: {best_config[col]:.0f}mm")
                        
                        # Option to apply the best configuration
                        if st.button("Apply Best Configuration"):
                            # Update thicknesses
                            for idx, thickness in zip([i for i, _ in thickness_ranges], best_config[:-1]):
                                material = st.session_state.layers[idx]["material"]
                                key = f"{material} (mm)"
                                st.session_state.layers[idx]["thickness"] = int(best_config[key])
                            
                            st.success("Applied the best configuration to your wall design!")
                            st.rerun()  # Updated from st.experimental_rerun()
                    
                    # Display full results table
                    with st.expander("View All Results"):
                        st.dataframe(
                            results_df.sort_values("U-Value (W/m²K)").reset_index(drop=True),
                            use_container_width=True
                        )

# Information Tab
with tab3:
    st.markdown("### U-Value Information")
    
    st.markdown(r"""
    #### What is a U-Value?
    
    A U-value (thermal transmittance) measures how effective a material is as an insulator.
    It represents the rate of heat transfer through a structure (like a wall, roof, or window), divided by the difference in temperature across that structure.
    
    The U-value is expressed in watts per square meter kelvin (W/m²K).
    
    #### How is it calculated?
    
    The U-value is calculated as the reciprocal of the total thermal resistance of the wall:
    
    $$U = \frac{1}{R_{total}}$$
    
    Where $R_{total}$ is the sum of all thermal resistances:
    
    $$R_{total} = R_{si} + \sum_{i=1}^{n} R_i + R_{se}$$
    
    - $R_{si}$ is the internal surface resistance
    - $R_{se}$ is the external surface resistance
    - $R_i$ is the thermal resistance of each material layer
    
    For each material layer, the thermal resistance is:
    
    $$R_i = \frac{d_i}{\lambda_i}$$
    
    Where:
    - $d_i$ is the thickness of the layer in meters
    - $\lambda_i$ is the thermal conductivity of the material in W/mK
    
    #### Typical U-value targets
    
    | Building Standard | Wall U-value (W/m²K) |
    |-------------------|----------------------|
    | Building Regulations (UK) | 0.30 |
    | Low Energy Building | 0.20 |
    | Passive House | 0.15 |
    
    Lower U-values indicate better thermal insulation.
    """)
    
    st.markdown("### Materials Database")
    
    # Display materials database grouped by category
    st.dataframe(
        materials_df.sort_values(["Category", "Material"]),
        column_config={
            "Category": st.column_config.TextColumn("Category"),
            "Material": st.column_config.TextColumn("Material"),
            "R-Value (m²K/W)": st.column_config.NumberColumn("R-Value (m²K/W)", format="%.3f"),
            "Color": st.column_config.TextColumn("Color"),
        },
        hide_index=True,
        use_container_width=True
    )
