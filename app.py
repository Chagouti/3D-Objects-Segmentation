
import streamlit as st
import numpy as np
import os
import subprocess
import plotly.graph_objects as go

def load_ad_file(ad_file_path):
    points = []
    normals = []
    labels = []

    with open(ad_file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 7:  # 3 for point, 3 for normal, 1 for label
                point = [float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])]
                normal = [float(parts[3]), float(parts[4]), float(parts[5])]
                label = int(parts[6])
                points.append(point)
                normals.append(normal)
                labels.append(label)

    return np.array(points), np.array(normals), np.array(labels)

def save_to_npy(point_cloud, file_path):
    """Save point cloud to .npy file."""
    np.save(file_path, point_cloud)

def run_classification_script():
    output = subprocess.run(
        ["python", "predict_single_object.py", "--use_normals", "--log_dir", "pointnet2_ssg_wo_normals"],
        cwd="./",
        capture_output=True,
        text=True
    )
    # The classification script now returns the predicted class directly (0 or 1)
    if int(output.stdout.strip())==0:
      return "vessel"
    else:
      return "aneurysm"
  

def plot_point_cloud(points):
    """Plot 3D point cloud using Plotly."""
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color=points[:, 2], colorscale='Viridis', opacity=0.8)
    )])
    fig.update_layout(scene=dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Z'
    ))
    st.plotly_chart(fig)

def main():
    st.title("3D Object Classification")

    # Sidebar for file upload
    st.sidebar.title("Upload AD File")
    uploaded_file = st.sidebar.file_uploader("Upload your AD file", type=["ad"])

    if uploaded_file is not None:
        # Process the uploaded AD file with load_file
        with open("uploaded_object.ad", "wb") as f:
            f.write(uploaded_file.getbuffer())

        point_cloud, normals, labels = load_ad_file('uploaded_object.ad')

        # 3D visualization
        st.subheader("3D Visualization of Point Cloud")
        plot_point_cloud(point_cloud)
        
        # Sample points if there are more than num_points
        if len(point_cloud) > 1024:
             indices = np.random.choice(len(point_cloud), 1024, replace=False)
             point_cloud = point_cloud[indices]

        # Randomly choose points with replacement to ensure num_points samples
        if len(point_cloud)<1024:
             choice = np.random.choice(len(point_cloud), 1024, replace=True)
             # Sample the selected points from the point_set
             point_cloud = point_cloud[choice, :]

        point_cloud = point_cloud[np.newaxis, :, :]  # Convert to (1, 1024, 6)
        label = [1 if 1 in labels else 0]

        # Save the processed point cloud to a .npy file
        npy_point_path = 'all_points.npy'
        npy_label_path = 'all_labels.npy'
        save_to_npy(point_cloud, npy_point_path)
        save_to_npy(label, npy_label_path)



        # Create a placeholder for loading message or spinner
        placeholder = st.empty()

        # Display loading message
        placeholder.text("Running the classification script, please wait...")

        # Run the classification script
        predicted_class = run_classification_script()

        # Clear the placeholder (remove the loading message)
        placeholder.empty()

        # Display the predicted class
        if predicted_class is not None:
            st.success(f"Predicted Class: {predicted_class}")
        else:
            st.error("Could not get predicted class.")

if __name__ == "__main__":
    main()
