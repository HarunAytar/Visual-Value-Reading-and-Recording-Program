import os
import easyocr
from PIL import Image
import numpy as np 
import datetime
import pandas as pd
from ultralytics import YOLO
import customtkinter as ctk
from tkinter import filedialog, messagebox


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

current_dir = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(current_dir, "kaydedilen_model.pt")

CLASS_NAMES = [
    'mevcut hız', 'tam hız', '1. bıçak basıncı', '2. bıçak basıncı', '3. bıçak basıncı', 
    '4. bıçak basıncı', '5. bıçak basıncı', '6. bıçak basıncı', '7. bıçak basıncı', '8. bıçak basıncı', 
    '1. hedef viskozite', '2. hedef viskozite', '3. hedef viskozite', '4. hedef viskozite', '5. hedef viskozite', 
    '6. hedef viskozite', '7. hedef viskozite', '8. hedef viskozite', '1. mevcut viskozite', '2. mevcut viskozite', 
    '3. mevcut viskozite', '4. mevcut viskozite', '5. mevcut viskozite', '6. mevcut viskozite', '7. mevcut viskozite', 
    '8. mevcut viskozite', 'tambur sıcaklığı', 'kurutma sıcaklığı', '1. baskı düzeltme', '3. baskı düzeltme', 
    '4. baskı düzeltme', '5. baskı düzeltme', '8. baskı düzeltme', '6. FS-TT', '6. FS-KT', '6. TM-TT', 
    '6. TM-KT', '7. FS-TT', '7. FS-KT', '7. TM-TT', '7. TM-KT', '6. baskı düzeltme', '7. baskı düzeltme', 
    '1. FS-TT', '1. FS-KT', '1. TM-TT', '1. TM-KT', '3. FS-TT', '3. FS-KT', '3. TM-TT', '3. TM-KT', 
    '4. FS-TT', '4. FS-KT', '4. TM-TT', '4. TM-KT', '5. FS-TT', '5. FS-KT', '5. TM-TT', '5. TM-KT', 
    '8. FS-TT', '8. FS-KT', '8. TM-TT', '8. TM-KT', '2. baskı düzeltme', '2. FS-TT', '2. FS-KT', 
    '2. TM-TT', '2. TM-KT'
]

class AyarOkuyucuApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Makine Ayar Değeri Okuyucu v2.0")
        self.geometry("800x700")
        self.image_path = None
        
        
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model Hatası", f"YOLO model dosyası yüklenemedi: {e}")
            self.model = None

        self.reader = easyocr.Reader(['tr', 'en'], verbose=False)
        self.create_widgets()

    def create_widgets(self):
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        
        self.header_frame = ctk.CTkFrame(self)
        self.header_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        self.path_label = ctk.CTkLabel(self.header_frame, text="Lütfen bir JPG dosyası seçin", font=("Segoe UI", 12))
        self.path_label.pack(pady=10)

        self.btn_frame = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        self.btn_frame.pack(pady=10)

        self.select_button = ctk.CTkButton(self.btn_frame, text="Dosya Seç", command=self.select_file, width=150)
        self.select_button.grid(row=0, column=0, padx=10)

        self.process_button = ctk.CTkButton(self.btn_frame, text="İşlemi Başlat", command=self.process_data, 
                                            state="disabled", fg_color="#2c3e50", width=150)
        self.process_button.grid(row=0, column=1, padx=10)

        
        self.output_label = ctk.CTkLabel(self, text="Okuma Sonuçları", font=("Segoe UI", 14, "bold"))
        self.output_label.grid(row=1, column=0, padx=20, sticky="w")

        self.output_text = ctk.CTkTextbox(self, font=("Consolas", 13), border_width=2)
        self.output_text.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="nsew")
        
        
        self.status_label = ctk.CTkLabel(self, text="Sistem Hazır", font=("Segoe UI", 11), anchor="w")
        self.status_label.grid(row=3, column=0, padx=20, pady=5, sticky="ew")

    def select_file(self):
        file_path = filedialog.askopenfilename(
            title="Görsel Seçin",
            filetypes=(("Görsel Dosyaları", "*.jpg *.jpeg"),)
        )
        if file_path:
            self.image_path = file_path
            self.path_label.configure(text=f"Seçilen: {os.path.basename(file_path)}", text_color="#3498db")
            self.process_button.configure(state="normal", fg_color="#27ae60")
            self.status_label.configure(text="Dosya yüklendi, işleme hazır.")
            self.output_text.delete("1.0", "end")

    def update_output(self, message):
        self.output_text.insert("end", message + "\n")
        self.output_text.see("end")
        self.update_idletasks()

    def process_data(self):
        if not self.image_path or not self.model: return

        self.output_text.delete("1.0", "end")
        self.status_label.configure(text="İşleniyor... Lütfen bekleyin.")
        self.update_output(">>> İŞLEM BAŞLATILDI")

        try:
            file_name, modification_time = self._get_file_info(self.image_path)
            self.update_output(f"📅 Tarih: {modification_time}")
            
            results = self.model.predict(source=self.image_path, conf=0.25, iou=0.0, save=False, verbose=False)
            
            if not results or not results[0].boxes:
                self.update_output("⚠️ HATA: Nesne tespit edilemedi.")
                return
            
            original_img = Image.open(self.image_path)
            img_width, img_height = original_img.size
            output_data = []

            for i, box in enumerate(results[0].boxes):
                coords = box.xywhn[0].cpu().numpy()
                class_id = int(box.cls[0].item())
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"ID: {class_id}"

                center_x, center_y, width, height = coords
                left = int((center_x - width/2) * img_width)
                top = int((center_y - height/2) * img_height)
                right = int((center_x + width/2) * img_width)
                bottom = int((center_y + height/2) * img_height)

                cropped_img = original_img.crop((left, top, right, bottom))
                ocr_results = self.reader.readtext(np.array(cropped_img))
                read_value = ocr_results[0][1] if ocr_results else "---"
                
                self.update_output(f"📍 {class_name.upper():<20} : {read_value}")

                output_data.append({
                    'Dosya Adı': file_name, 'Çekim Tarihi': modification_time,
                    'Ayar Adı': class_name, 'Okunan Değer': read_value, 'Sınıf ID': class_id
                })
            
            self._save_to_excel(output_data)

        except Exception as e:
            self.update_output(f"\n❌ KRİTİK HATA: {e}")
        finally:
            self.status_label.configure(text="İşlem Tamamlandı.")

    def _get_file_info(self, file_path):
        stat_info = os.stat(file_path)
        mod_time = datetime.datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        return os.path.basename(file_path), mod_time

    def _save_to_excel(self, output_data):
        if not output_data: return
        df = pd.DataFrame(output_data)
        path = os.path.join(os.getcwd(), 'ayar_degerleri_ciktisi.xlsx')
        
        try:
            if os.path.exists(path):
                df_existing = pd.read_excel(path)
                df = pd.concat([df_existing, df], ignore_index=True)
            df.to_excel(path, index=False)
            self.update_output(f"\n✅ Excel'e kaydedildi: {os.path.basename(path)}")
        except Exception as e:
            self.update_output(f"Excel hatası: {e}")

if __name__ == "__main__":
    app = AyarOkuyucuApp()
    app.mainloop()