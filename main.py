
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# =====================================
# 1. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Gear Parameter Calculator",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Gear Parameter Calculator")
st.subheader("Calculation of Gear Parameters Using Machine Vision")

st.write(
    "Upload a spur gear image to analyze its geometry "
    "and calculate important gear parameters."
)

st.divider()

# =====================================
# 2. IMAGE UPLOAD
# =====================================

st.header("1. Upload Gear Image")

uploaded_file = st.file_uploader(
    "Choose a gear image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)

    st.success("Image uploaded successfully!")

    # =================================
    # 3. GRAYSCALE CONVERSION
    # =================================

    gray_image = cv2.cvtColor(
        image_array,
        cv2.COLOR_RGB2GRAY
    )

    # =================================
    # 4. GAUSSIAN BLUR
    # =================================

    blurred = cv2.GaussianBlur(
        gray_image,
        (5, 5),
        0
    )

    # =================================
    # 5. EDGE DETECTION
    # =================================

    edges = cv2.Canny(
        blurred,
        50,
        150
    )

    # =================================
    # 6. DISPLAY PROCESSED IMAGES
    # =================================

    st.divider()
    st.header("2. Image Processing Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Gear Image")
        st.image(image_array, use_container_width=True)

    with col2:
        st.subheader("Grayscale Image")
        st.image(
            gray_image,
            clamp=True,
            use_container_width=True
        )

    st.subheader("3. Gear Edge Detection")

    st.image(
        edges,
        caption="Detected Gear Edges",
        clamp=True,
        use_container_width=True
    )

    # =================================
    # 7. AUTOMATIC GEAR TOOTH COUNTING
    # =================================

    st.divider()
    st.header("4. Automatic Gear Tooth Counting")

    _, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    if contours:

        gear_contour = max(
            contours,
            key=cv2.contourArea
        )

        (cx, cy), radius = cv2.minEnclosingCircle(
            gear_contour
        )

        points = gear_contour.reshape(-1, 2)

        dx = points[:, 0] - cx
        dy = points[:, 1] - cy

        angles = np.mod(
            np.arctan2(dy, dx),
            2 * np.pi
        )

        distances = np.sqrt(dx**2 + dy**2)

        bins = 720
        radial_profile = np.zeros(bins)

        indices = (
            angles * bins / (2 * np.pi)
        ).astype(int)

        indices = np.clip(indices, 0, bins - 1)

        np.maximum.at(
            radial_profile,
            indices,
            distances
        )

        valid = radial_profile > 0

        if np.count_nonzero(valid) > 10:

            x = np.arange(bins)

            radial_profile = np.interp(
                x,
                x[valid],
                radial_profile[valid]
            )

            smooth_profile = np.convolve(
                np.pad(
                    radial_profile,
                    (3, 3),
                    mode="wrap"
                ),
                np.ones(7) / 7,
                mode="valid"
            )

            threshold = (
                np.mean(smooth_profile)
                + 0.30 * (
                    np.max(smooth_profile)
                    - np.mean(smooth_profile)
                )
            )

            peaks = []

            for i in range(bins):

                previous = smooth_profile[(i - 1) % bins]
                current = smooth_profile[i]
                following = smooth_profile[(i + 1) % bins]

                if (
                    current > previous
                    and current >= following
                    and current > threshold
                ):
                    peaks.append(i)

            selected_peaks = []

            for peak in sorted(
                peaks,
                key=lambda p: smooth_profile[p],
                reverse=True
            ):

                if all(
                    min(
                        abs(peak - p),
                        bins - abs(peak - p)
                    ) >= 16
                    for p in selected_peaks
                ):
                    selected_peaks.append(peak)

            tooth_count = len(selected_peaks)

            # Display tooth count

            st.metric(
                "Estimated Number of Gear Teeth",
                tooth_count
            )

            # Display detected contour

            result_image = image_array.copy()

            cv2.drawContours(
                result_image,
                [gear_contour],
                -1,
                (0, 255, 0),
                2
            )

            cv2.circle(
                result_image,
                (int(cx), int(cy)),
                5,
                (255, 0, 0),
                -1
            )

            st.image(
                result_image,
                caption="Detected Gear Boundary and Center",
                use_container_width=True
            )

            st.warning(
                "Verify the estimated tooth count "
                "against the actual gear."
            )

            # =================================
            # 8. GEAR PARAMETER CALCULATION
            # =================================

            st.divider()
            st.header("5. Gear Parameter Calculation")

            st.write(
                "Enter the measured outside diameter "
                "of your gear in millimetres."
            )

            outside_diameter = st.number_input(
                "Enter Outside Diameter (mm)",
                min_value=0.1,
                value=44.0,
                step=0.5
            )

            actual_teeth = st.number_input(
                "Number of Teeth (verify or correct)",
                min_value=3,
                value=max(3, int(tooth_count)),
                step=1
            )

            if st.button("Calculate Gear Parameters"):

                Z = int(actual_teeth)
                Da = float(outside_diameter)

                # Standard full-depth spur gear formulas

                module = Da / (Z + 2)

                pitch_diameter = module * Z

                root_diameter = module * (Z - 2.5)

                addendum = module

                dedendum = 1.25 * module

                # Display calculated results

                st.success(
                    "Gear parameters calculated successfully!"
                )

                st.subheader("Calculated Gear Parameters")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Module",
                    f"{module:.2f} mm"
                )

                col2.metric(
                    "Pitch Diameter",
                    f"{pitch_diameter:.2f} mm"
                )

                col3.metric(
                    "Root Diameter",
                    f"{root_diameter:.2f} mm"
                )

                col4, col5, col6 = st.columns(3)

                col4.metric(
                    "Outside Diameter",
                    f"{Da:.2f} mm"
                )

                col5.metric(
                    "Addendum",
                    f"{addendum:.2f} mm"
                )

                col6.metric(
                    "Dedendum",
                    f"{dedendum:.2f} mm"
                )

                # Results table

                results = {
                    "Gear Parameter": [
                        "Number of Teeth",
                        "Module",
                        "Outside Diameter",
                        "Pitch Circle Diameter",
                        "Root Diameter",
                        "Addendum",
                        "Dedendum"
                    ],

                    "Calculated Value": [
                        str(Z),
                        f"{module:.2f} mm",
                        f"{Da:.2f} mm",
                        f"{pitch_diameter:.2f} mm",
                        f"{root_diameter:.2f} mm",
                        f"{addendum:.2f} mm",
                        f"{dedendum:.2f} mm"
                    ]
                }

                df = pd.DataFrame(results)

                st.subheader("Gear Parameter Results")

                st.table(df)

                # Download results

                csv = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Results as CSV",
                    data=csv,
                    file_name="gear_parameters.csv",
                    mime="text/csv"
                )

        else:
            st.error(
                "Not enough gear boundary points detected."
            )

    else:
        st.error(
            "No gear contour detected. "
            "Try a clear top-view image."
        )

else:
    st.info("Please upload a spur gear image to begin.")

# =====================================
# 9. PROJECT FOOTER
# =====================================

st.divider()

st.caption(
    "Gear Parameter Calculator | Python | OpenCV | Streamlit"
)
