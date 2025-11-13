import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os
import random

# ===============================
# Judul Aplikasi
# ===============================
st.title("🍽️ Klasifikasi Gambar Makanan dengan YOLOv8")

# ===============================
# Upload Gambar
# ===============================
uploaded_file = st.file_uploader("📤 Pilih gambar makanan...", type=["jpg", "jpeg", "png"])

# ===============================
# Load Model
# ===============================
model_path = "food_classifier_yolov8n_cls.pt"  # ubah sesuai model kamu
model = YOLO(model_path)

# ===============================
# Lokasi Dataset
# ===============================
dataset_path = "dataset"  # ubah sesuai lokasi dataset kamu

# ===============================
# Proses Jika Ada File Diupload
# ===============================
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")  # pastikan format valid
        st.image(image, caption="📷 Gambar Asli", width='stretch')

        # Simpan sementara
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image_path = tmp.name
            image.save(image_path)

        # Jalankan prediksi
        st.write("⏳ Sedang memproses gambar...")
        results = model(image_path)

        # Gambar hasil prediksi (klasifikasi)
        result_image = results[0].plot()
        st.image(result_image, caption="🧠 Hasil Prediksi Model", width='stretch')

        # ===============================
        # Ambil Label Prediksi
        # ===============================
        if hasattr(results[0], "probs") and results[0].probs is not None:
            probs = results[0].probs.data.cpu().numpy()
            top_index = probs.argmax()
            label = model.names[top_index]
            conf = probs[top_index]
            st.subheader(f"📋 Hasil Prediksi: **{label} ({conf:.2f})**")

            # ===============================
            # Tampilkan 10 Gambar dari Dataset Kelas yang Sama
            # ===============================
            class_folder = os.path.join(dataset_path, label)
            if os.path.exists(class_folder):
                images = [os.path.join(class_folder, f) for f in os.listdir(class_folder)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                if len(images) > 0:
                    st.write(f"🖼️ Contoh gambar lain dari kelas **{label}**:")
                    sample_images = random.sample(images, min(10, len(images)))

                    cols = st.columns(5)
                    for idx, img_path in enumerate(sample_images):
                        with cols[idx % 5]:
                            st.image(img_path, width='stretch')
                else:
                    st.warning("⚠️ Tidak ada gambar contoh pada folder kelas ini.")
            else:
                st.error(f"❌ Folder kelas '{label}' tidak ditemukan di {dataset_path}")
        else:
            st.error("❌ Model tidak menghasilkan probabilitas (kemungkinan model deteksi, bukan klasifikasi).")

    except Exception as e:
        st.error(f"Gagal membuka atau memproses gambar: {e}")

else:
    st.info("⬆️ Silakan upload gambar terlebih dahulu untuk mulai klasifikasi.")
