
import streamlit as st
import cv2
import numpy as np
import os
import pandas as pd

st.title(" Zone Detector – SE Product Counter")

# -----------------------------
# Folder Selection
# -----------------------------
input_folder = st.text_input("Select Input Folder")
output_folder = st.text_input("Select Output Folder")

# -----------------------------
# Run Button
# -----------------------------
if st.button("Run Detection"):
    if not os.path.isdir(input_folder):
        st.error("❌ Input folder does not exist")
    elif not os.path.isdir(output_folder):
        st.error("❌ Output folder does not exist")
    else:
        st.success("Processing started…")

        os.makedirs(output_folder, exist_ok=True)
        results = []

        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        for filename in os.listdir(input_folder):
            if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(input_folder, filename)
            img = cv2.imread(path)

            if img is None:
                continue

            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_green, upper_green)

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            green_count = sum(1 for cnt in contours if cv2.contourArea(cnt) >= 500)
            half_green = (green_count // 2) + (green_count % 2)

            results.append({"Image": filename, "SE Products": half_green})

            # Draw results
            for cnt in contours:
                if cv2.contourArea(cnt) < 500:
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(img, "SE", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            cv2.putText(img, f"SE Products: {half_green}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

            cv2.imwrite(os.path.join(output_folder, filename), img)

        # Export Excel
        df = pd.DataFrame(results)
        excel_path = os.path.join(output_folder, "SE_checked_results.xlsx")
        df.to_excel(excel_path, index=False)

        st.success("✅ Processing complete!")
        st.write("Excel saved at:", excel_path)
        st.dataframe(df)
