import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from PySide6.QtWidgets import QVBoxLayout, QWidget
from vtk.util.numpy_support import numpy_to_vtk
import numpy as np


class CustomInteractorStyle(vtk.vtkInteractorStyleImage):
    def __init__(self, viewer):
        super().__init__()  # Correct initialization of the base class
        self.viewer = viewer

        # Observers for mouse wheel events to handle slice scrolling
        self.AddObserver("MouseWheelForwardEvent", self.scroll_slices_forward)
        self.AddObserver("MouseWheelBackwardEvent", self.scroll_slices_backward)
        self.AddObserver("RightButtonPressEvent", self.right_button_press)
        self.AddObserver("RightButtonReleaseEvent", self.right_button_release)
        self.AddObserver("LeftButtonPressEvent", self.left_button_press)
        self.AddObserver("LeftButtonReleaseEvent", self.left_button_release)

        self.panning = False
        self.zooming = False
        self.last_position = None

    def scroll_slices_forward(self, obj, event):
        # Scroll slices forward instead of zooming
        self.viewer.scroll_slices(self.viewer.current_slice + 1)

        # Override zoom behavior by not calling the default method
        return

    def scroll_slices_backward(self, obj, event):
        # Scroll slices backward instead of zooming
        self.viewer.scroll_slices(self.viewer.current_slice - 1)

        # Override zoom behavior by not calling the default method
        return

    def right_button_press(self, obj, event):
        self.zooming = True
        self.OnRightButtonDown()

    def right_button_release(self, obj, event):
        self.zooming = False
        self.OnRightButtonUp()

    def left_button_press(self, obj, event):
        self.panning = True
        self.OnLeftButtonDown()

    def left_button_release(self, obj, event):
        self.panning = False
        self.OnLeftButtonUp()

    def OnMouseMove(self, obj, event):
        # Custom behavior on mouse move
        interactor = self.GetInteractor()
        if self.zooming:
            self.OnMouseWheelForward()
        elif self.panning:
            self.OnMouseMove()
        else:
            super().OnMouseMove()  # Call the parent class method to handle other mouse movements

        # Update rendering
        self.viewer.render_slice()


class Viewer(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.image_reader = None
        self.image_data = None
        self.rtstruct_reader = None
        self.rtdose_reader = None
        self.dose_overlay = None
        self.current_slice = 0
        self.show_dose_overlay = False
        self.dose_units = "Gy"  # Default to Gy if not specified in the file
        self.dose_threshold = 0.0  # Initial dose threshold
        self.window_ = 400
        self.level = 50
        self.structure_checkboxes = {}
        self.visible_structures = {}  # To keep track of which structures are visible
        self.scalar_bar = None
        self.structure_masks = {}
        self.lookup_table = None

        # Variables to store the camera state
        self.camera_position = None
        self.camera_focal_point = None
        self.camera_view_up = None
        self.camera_parallel_scale = None
        self.camera_initialized = False

    def init_ui(self):
        layout = QVBoxLayout()

        # Create VTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget)

        self.setLayout(layout)

        # Set up VTK rendering
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
        self.vtk_interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Create and set the custom interaction style
        self.interactor_style = CustomInteractorStyle(self)
        self.vtk_interactor.SetInteractorStyle(self.interactor_style)

        self.vtk_renderer.SetBackground(0.0, 0.0, 0.0)
        self.vtk_interactor.Initialize()

    def display_image(self, image_array):
        """Display the image array in the viewer."""
        if image_array is None or image_array.size == 0:
            print("Error: Image array is empty or None.")
            return

        # Set default window and level based on modality
        if self.image_reader:
            modality = self.image_reader.modality
            self.set_default_window_level(modality)

        self.image_data = image_array
        self.current_slice = len(image_array) // 2  # Start in the middle

        self.render_slice()

        # Reset camera for the new image
        self.camera_initialized = False
        self.fit_image_to_view()

        self.vtk_widget.GetRenderWindow().Render()

    def render_slice(self):
        if self.image_data is None:
            return

        # Store the window and level from the current image actor if it exists
        if hasattr(self, "image_actor") and self.image_actor:
            self.window_ = self.image_actor.GetProperty().GetColorWindow()
            self.level = self.image_actor.GetProperty().GetColorLevel()

        # Ensure that the renderer and actors are properly removed before adding new one
        self.vtk_renderer.RemoveAllViewProps()

        # Check for OpenGL context-related issues before rendering
        render_window = self.vtk_widget.GetRenderWindow()
        render_window.MakeCurrent()

        slice_data = self.image_data[self.current_slice, :, :]

        vtk_image_data = vtk.vtkImageData()
        vtk_image_data.SetDimensions(slice_data.shape[1], slice_data.shape[0], 1)
        vtk_image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        vtk_array = numpy_to_vtk(slice_data.ravel(), deep=True, array_type=vtk.VTK_FLOAT)
        vtk_image_data.GetPointData().SetScalars(vtk_array)

        # Flip the image vertically using vtkImageFlip
        flip_filter = vtk.vtkImageFlip()
        flip_filter.SetFilteredAxis(1)  # Flip along the y-axis
        flip_filter.SetInputData(vtk_image_data)
        flip_filter.Update()

        image_actor = vtk.vtkImageActor()
        image_actor.GetMapper().SetInputData(flip_filter.GetOutput())

        # Set window and level
        image_actor.GetProperty().SetColorWindow(self.window_)
        image_actor.GetProperty().SetColorLevel(self.level)

        self.vtk_renderer.RemoveAllViewProps()
        self.vtk_renderer.AddActor(image_actor)

        if self.show_dose_overlay and self.dose_overlay is not None:
            self.render_dose_overlay()

        self.render_structures()

        # Ensure the camera state is applied correctly afte the image is rendered
        if not self.camera_initialized:
            self.fit_image_to_view()
        else:
            self.apply_camera_state()

        # # Apply the saved camera state to maintain zoom and position
        # if self.camera_initialized:
        #     self.apply_camera_state()
        # else:
        #     self.vtk_renderer.ResetCamera()

        # Re-add scalar bar after rendering
        if self.scalar_bar and not self.vtk_renderer.HasViewProp(self.scalar_bar):
            self.vtk_renderer.AddActor2D(self.scalar_bar)

        self.vtk_widget.GetRenderWindow().Render()

    def fit_image_to_view(self):
        """Adjust the camera zoom to fit the image to the viewing area."""
        self.vtk_renderer.ResetCamera()  # Reset camera to default settings
        self.vtk_renderer.ResetCameraClippingRange()

        # Get the renderer's camera
        camera = self.vtk_renderer.GetActiveCamera()

        # Get the size of the image and the rendering window
        view_size = self.vtk_widget.size()
        view_width, view_height = view_size.width(), view_size.height()

        image_width, image_height = self.image_data.shape[2], self.image_data.shape[1]

        # Calculate the scaling factor to fit the image in the view
        scale_width = view_width / image_width
        scale_height = view_height / image_height
        scale = min(scale_width, scale_height)

        # Adjust the parallel scale of the camera to zoom correctly
        camera.SetParallelScale(image_height / 2.0)
        camera.Zoom(scale)

        # Force an update to the camera settings
        self.vtk_renderer.ResetCamera()
        self.vtk_renderer.ResetCameraClippingRange()

        # Save the camera state after initial fitting
        self.save_camera_state()
        self.camera_initialized = True

    def save_camera_state(self):
        """Save the current camera state (position, focal point, etc.)."""
        camera = self.vtk_renderer.GetActiveCamera()
        self.camera_position = camera.GetPosition()
        self.camera_focal_point = camera.GetFocalPoint()
        self.camera_view_up = camera.GetViewUp()
        self.camera_parallel_scale = camera.GetParallelScale()

    def apply_camera_state(self):
        """Apply the saved camera state (position, focal point, etc.)."""
        if self.camera_position and self.camera_focal_point and self.camera_view_up:
            camera = self.vtk_renderer.GetActiveCamera()
            camera.SetPosition(self.camera_position)
            camera.SetFocalPoint(self.camera_focal_point)
            camera.SetViewUp(self.camera_view_up)
            camera.SetParallelScale(self.camera_parallel_scale)
        else:
            self.vtk_renderer.ResetCamera()

    def overlay_rtstruct(self, rtstruct_reader):
        """Overlay RTSTRUCT contours on the image."""
        self.rtstruct_reader = rtstruct_reader
        self.visible_structures.clear()
        for structure_name in rtstruct_reader.get_structure_names():
            self.visible_structures[structure_name] = False  # Start with all contours hidden

        self.render_slice()

    def overlay_dose(self, dose_data):
        """Overlay dose data on the image."""
        self.dose_overlay = dose_data
        self.show_dose_overlay = True

        # self.dose_threshold = dose_data.max() / 2

        # Create a lookup table for dose visualization
        self.lookup_table = vtk.vtkLookupTable()
        self.lookup_table.SetNumberOfTableValues(1024)
        self.lookup_table.SetRange(self.dose_threshold, dose_data.max())
        self.lookup_table.Build()

        # Define a more detailed color mapping with more intermediate colors
        colors = [
            (0.0, 0.0, 0.5),  # Dark Blue
            (0.0, 0.0, 1.0),  # Blue
            (0.0, 0.5, 1.0),  # Light Blue
            (0.0, 1.0, 1.0),  # Cyan
            (0.0, 1.0, 0.5),  # Greenish Cyan
            (0.0, 1.0, 0.0),  # Green
            (0.5, 1.0, 0.0),  # Yellowish Green
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.75, 0.0),  # Orange-Yellow
            (1.0, 0.5, 0.0),  # Orange
            (1.0, 0.25, 0.0),  # Reddish Orange
            (1.0, 0.0, 0.0),  # Bright Red
        ]

        # Interpolate the colors smoothly across the lookup table
        num_colors = len(colors)
        for i in range(self.lookup_table.GetNumberOfTableValues()):
            ratio = i / (self.lookup_table.GetNumberOfTableValues() - 1)
            low_color_index = int(ratio * (num_colors - 1))
            high_color_index = min(low_color_index + 1, num_colors - 1)
            interp_ratio = (ratio * (num_colors - 1)) - low_color_index

            # Linear interpolation between low_color and high_color
            low_color = colors[low_color_index]
            high_color = colors[high_color_index]
            interpolated_color = tuple(
                low_color[j] + interp_ratio * (high_color[j] - low_color[j]) for j in range(3)
            )

            # If dose value is zero, make it fully transparent
            alpha = 0.0 if ratio == 0 else 0.5

            self.lookup_table.SetTableValue(
                i, interpolated_color[0], interpolated_color[1], interpolated_color[2], alpha
            )

        self.render_slice()

        # Add dose legend
        self.add_dose_legend(self.lookup_table)

    def add_dose_legend(self, lookup_table):
        """Adds a scalar bar to represent dose levels."""
        if self.scalar_bar:
            self.vtk_renderer.RemoveActor(self.scalar_bar)

        # Create a scalar bar (legend) to show the dose levels
        self.scalar_bar = vtk.vtkScalarBarActor()
        self.scalar_bar.SetLookupTable(lookup_table)
        self.scalar_bar.SetTitle(f"{self.dose_units}           ")
        self.scalar_bar.SetNumberOfLabels(5)

        # Adjust the size and position of the scalar bar (legend)
        self.scalar_bar.SetWidth(0.05)  # Set width relative to the window size
        self.scalar_bar.SetHeight(0.3)  # Set height relative to the window size
        self.scalar_bar.SetPosition(0.9, 0.1)  # Position on the right side

        self.vtk_renderer.AddActor2D(self.scalar_bar)
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_structures(self, structure_name, visible):
        """Toggle the visibility of a specific structure."""
        self.visible_structures[structure_name] = visible
        self.render_slice()

    def set_window(self, window):
        self.window_ = window
        self.save_window_level()  # Save changes when window is set
        if self.image_data is not None:
            self.render_slice()

    def set_level(self, level):
        self.level = level
        self.save_window_level()  # Save changes when level is set
        if self.image_data is not None:
            self.render_slice()

    def set_default_window_level(self, modality):
        """Sets default window and level based on image modality."""
        if modality == "CT":
            self.window_ = 400
            self.level = 40
        elif modality == "MR":
            self.window_, self.level = self.get_window_level_from_mri_sequence(
                self.image_reader.image
            )

        elif modality == "PT":
            self.window_ = 600
            self.level = 300

        else:
            self.window_ = 255
            self.level = 127

    def render_structures(self):
        """Render all visible structures (contours) in the viewer."""
        if not self.visible_structures or not self.rtstruct_reader:
            return

        for structure_name, is_visible in self.visible_structures.items():
            if is_visible:
                try:
                    contours_in_pixel_space = (
                        self.rtstruct_reader.get_structure_contour_points_in_pixel_space(
                            structure_name, self.image_reader
                        )
                    )
                    color = self.rtstruct_reader.get_structure_color(structure_name)
                    self.add_structure_contours_to_renderer(contours_in_pixel_space, color)
                except Exception as e:
                    print(f"Error rendering structure {structure_name}: {e}")

        self.vtk_widget.GetRenderWindow().Render()

    def render_dose_overlay(self):
        if self.dose_overlay is None:
            return

        # Apply dose threshold
        thresholded_dose_slice = np.where(
            self.dose_overlay[self.current_slice, :, :] >= self.dose_threshold,
            self.dose_overlay[self.current_slice, :, :],
            0,
        )

        vtk_image_data = vtk.vtkImageData()
        vtk_image_data.SetDimensions(
            thresholded_dose_slice.shape[1], thresholded_dose_slice.shape[0], 1
        )
        vtk_image_data.AllocateScalars(vtk.VTK_FLOAT, 1)
        vtk_array = numpy_to_vtk(
            thresholded_dose_slice.ravel(), deep=True, array_type=vtk.VTK_FLOAT
        )
        vtk_image_data.GetPointData().SetScalars(vtk_array)

        # Flip the image vertically using vtkImageFlip
        flip_filter = vtk.vtkImageFlip()
        flip_filter.SetFilteredAxis(1)  # Flip along the y-axis
        flip_filter.SetInputData(vtk_image_data)
        flip_filter.Update()

        dose_color_mapper = vtk.vtkImageMapToColors()
        dose_color_mapper.SetInputData(flip_filter.GetOutput())
        dose_color_mapper.SetLookupTable(self.lookup_table)
        dose_color_mapper.Update()

        dose_actor = vtk.vtkImageActor()
        dose_actor.GetMapper().SetInputConnection(dose_color_mapper.GetOutputPort())
        self.vtk_renderer.AddActor(dose_actor)

        # # Update the legend to use the same lookup table
        # self.add_dose_legend(self.lookup_table)

    def add_structure_contours_to_renderer(self, contours_in_pixel_space, color):
        """
        Add the structure contours to the renderer for the current slice.

        Args:
            contours_in_pixel_space (dict): The contour points in pixel space, organized by
            slice index.
            color (list): The RGB color to use for the contours.
        """
        # Get the contours for the current slice
        if self.current_slice in contours_in_pixel_space:
            contours = contours_in_pixel_space[self.current_slice]

            # Process each contour in the current slice
            for contour in contours:
                # Create a vtkPoints object to store the contour points
                contour_points = vtk.vtkPoints()
                for point in contour:
                    # Insert contour points into vtkPoints (x, y, z=0)
                    contour_points.InsertNextPoint(point[1], point[0], 0)  # x, y, z=0

                # Create a vtkPolyLine to represent the contour
                contour_polyline = vtk.vtkPolyLine()
                contour_polyline.GetPointIds().SetNumberOfIds(len(contour))
                for i in range(len(contour)):
                    contour_polyline.GetPointIds().SetId(i, i)

                # Create a vtkCellArray to hold the polyline
                cells = vtk.vtkCellArray()
                cells.InsertNextCell(contour_polyline)

                # Create a vtkPolyData object to represent the contour data
                contour_data = vtk.vtkPolyData()
                contour_data.SetPoints(contour_points)
                contour_data.SetLines(cells)

                # Create a mapper and actor for the contour
                contour_mapper = vtk.vtkPolyDataMapper()
                contour_mapper.SetInputData(contour_data)

                contour_actor = vtk.vtkActor()
                contour_actor.SetMapper(contour_mapper)
                normalized_colors = [c / 255.0 for c in color]
                contour_actor.GetProperty().SetColor(*normalized_colors)
                contour_actor.GetProperty().SetLineWidth(2.0)

                # Add the actor to the renderer
                self.vtk_renderer.AddActor(contour_actor)

    def adjust_zoom(self, zoom_factor):
        """Adjust the zoom level of the renderer."""
        camera = self.vtk_renderer.GetActiveCamera()
        camera.Zoom(zoom_factor)
        self.vtk_widget.GetRenderWindow().Render()

    def scroll_slices(self, slice_index):
        """Scroll through the slices of the image data."""
        if self.image_data is not None:
            max_slice = self.image_data.shape[0] - 1
            self.current_slice = max(0, min(slice_index, max_slice))

            # Save the camera state before rendering the new slice
            self.save_camera_state()

            self.render_slice()

    # TODO:
    # implement this function
    # def display_dose(self, dose_data, dose_units):
    #     self.dose_overlay = dose_data
    #     self.dose_units = dose_units
    #     print(f"Dose overlay loaded with shape: {self.dose_overlay.shape}")
    #     self.render_slice()

    def toggle_dose_overlay(self, state):
        self.show_dose_overlay = state
        self.render_slice()

    def set_dose_threshold(self, threshold):
        """Set the dose threshold and update the lookup table accordingly."""
        self.dose_threshold = threshold

        if self.dose_overlay is not None:
            # Update the lookup table range dynamically
            min_dose = self.dose_threshold
            max_dose = self.dose_overlay.max()

            if self.lookup_table:
                self.lookup_table.SetRange(min_dose, max_dose)
                self.lookup_table.Build()

                # Re-render the slice to apply the new lookup table settings
                self.render_slice()

    def set_rtstruct_reader(self, rtstruct_reader):
        self.rtstruct_reader = rtstruct_reader

    def get_window_level_from_mri_sequence(self, image):
        # Extract relevant MR sequence information
        sequence_name = (
            image.GetMetaData("0018|0024") if image.HasMetaDataKey("0018|0024") else "Unknown"
        ).lower()
        pulse_sequence_name = (
            image.GetMetaData("0018|9005") if image.HasMetaDataKey("0018|9005") else "Unknown"
        ).lower()
        scanning_sequence = (
            image.GetMetaData("0018|0020") if image.HasMetaDataKey("0018|0020") else "Unknown"
        ).lower()

        series_description = {
            image.GetMetaData("0008|103e") if image.HasMetaDataKey("0008|103e") else "Unknown"
        }

        # Set default window and level
        window_width = 400
        window_level = 200

        # Determine appropriate window/level settings based on scanning sequence
        if (
            "t1" in sequence_name.lower()
            or "t1" in series_description
            or "t1" in pulse_sequence_name
            or "se" in scanning_sequence
        ):
            window_width = 400
            window_level = 100
        elif "t2" in series_description or "t2" in sequence_name or "t2" in pulse_sequence_name:
            window_width = 400
            window_level = 100
        elif (
            "flair" in series_description
            or "flair" in sequence_name
            or "flair" in pulse_sequence_name
        ):
            window_width = 350
            window_level = 90
        elif "pd" in series_description or "pd" in sequence_name or "pd" in pulse_sequence_name:
            window_width = 300
            window_level = 60
        elif "dwi" in series_description or "dwi" in sequence_name or "dwi" in pulse_sequence_name:
            window_width = 350
            window_level = 45
        elif "gre" in series_description or "gre" in sequence_name or "gre" in pulse_sequence_name:
            window_width = 150
            window_level = 50
        elif (
            "stir" in series_description
            or "stir" in sequence_name
            or "stir" in pulse_sequence_name
        ):
            window_width = 150
            window_level = 75

        return window_width, window_level

    def closeEvent(self, event):
        """Ensure proper cleanup when the widget or window is closed."""
        self.cleanup_vtk()

    def cleanup_vtk(self):
        """Clearn up VTK-related resources to prevent OpenGL context issues."""
        if self.vtk_widget:
            render_window = self.vtk_widget.GetRenderWindow()
            if render_window:
                render_window.Finalize()
                render_window.RemoveRenderer(self.vtk_renderer)
                render_window.SetInteractor(None)
        self.vtk_widget.deleteLater()
        self.vtk_renderer = None
