# VoiceDeff

Evaluasi proteksi suara berbasis PGD pada **LibriTTS** dan **VCTK**. Kode aktif sudah disatukan untuk kedua dataset. Hasil dengan protokol berbeda tetap dipisahkan agar interpretasinya jelas.

## Kesimpulan pemeriksaan

1. **Tidak ada dua folder hasil yang sepenuhnya identik.** Dari 144 file awal, terdapat 97 isi unik berdasarkan SHA-256. Semua isi awal dipertahankan; 47 salinan identik kini menunjuk ke satu penyimpanan fisik melalui hard link NTFS.
2. **Dua skrip `*-terbaru.py` dapat disatukan.** Perbedaannya hanya nama dataset, pola lokasi WAV, folder output, dan nama arsip. Kode aktif sekarang memakai satu konfigurasi dataset melalui CLI.
3. **Dua notebook static U-shape juga dapat disatukan.** Perbedaannya hanya `TARGET_DATASET` dan sumber hasil lama. Sebanyak **31 fungsi/kelas identik** dari skrip dan notebook dipindahkan ke `voice_deff/common.py`.
4. **Static U-shape bukan pengujian beberapa audio saja.** Notebook tersebut menghitung **satu metode baru** pada **320 probe × 3 seed = 960 baris**, kemudian memakai ulang enam metode lama. Skrip utama menghitung **enam metode × 320 probe × 3 seed = 5.760 baris**. Jadi yang berbeda adalah cakupan metode, bukan jumlah probe. Dataset sumber juga tidak diuji seluruhnya: keduanya memilih 500 utterance.
5. **Protokol evaluasi tidak boleh dicampur.** Folder `libri`/`vctk` menggunakan ambang ASV hasil kalibrasi dan statistik per speaker; folder `hasil_revisi_*` menggunakan ambang tetap 0,25 dan statistik antar-run. Keduanya disimpan sebagai versi terpisah. Skor dasar `s_asv`, PESQ, STOI, SSIM, dan SQUIM-MOS sama hingga toleransi 1e-10 pada pasangan baris yang sama; metrik keberhasilan berbasis ambang dapat berbeda.
6. **Hasil Static PGD + U-Shape belum tersedia dalam berkas awal.** Karena itu, efek faktorial dan keunggulan metode tambahan tersebut belum dapat disimpulkan dari hasil yang ada.

## Struktur

```text
VoiceDeff/
├── README.md
├── requirements.txt
├── voice_deff/
│   ├── __main__.py       # Satu CLI untuk dua dataset dan dua mode
│   ├── common.py         # 31 fungsi/kelas yang identik
│   ├── full.py           # Enam metode; ambang evaluasi tetap
│   └── factorial.py      # Tambahan Static U-Shape; memakai hasil terkalibrasi
├── results/
│   ├── libritts/
│   │   ├── calibrated_asv/
│   │   └── fixed_threshold/
│   └── vctk/
│       ├── calibrated_asv/
│       └── fixed_threshold/
├── archive/original_sources/ # Dua skrip dan dua notebook asli, tanpa modifikasi
├── reports/                 # Inventaris, pemetaan, duplikat, dan validasi
├── tools/                   # Audit dan skrip migrasi satu kali
└── runs/                    # Output baru saat eksperimen dijalankan
```

| Folder awal | Lokasi sekarang | Protokol |
|---|---|---|
| `libri/` | `results/libritts/calibrated_asv/` | Ambang ASV 0,854407; CI dari 40 speaker |
| `vctk/` | `results/vctk/calibrated_asv/` | Ambang ASV 0,847473; CI dari 40 speaker |
| `hasil_revisi_libritts/` | `results/libritts/fixed_threshold/` | Ambang ASV 0,25; CI dari 3 run |
| `hasil_revisi_vctk/` | `results/vctk/fixed_threshold/` | Ambang ASV 0,25; CI dari 3 run |

Pemilihan versi berdasarkan isi manifest, bukan berdasarkan nama folder atau dugaan urutan waktunya. Untuk analisis tambahan dari notebook yang tersedia, gunakan `calibrated_asv`. Untuk mereproduksi skrip terbaru yang diberikan, gunakan mode `full` dengan protokol ambang tetap.

## Instalasi dan data

Gunakan Python 3.10 atau lebih baru. Pasang PyTorch dan torchaudio dengan versi yang cocok untuk CPU/CUDA mesin, lalu:

```powershell
python -m pip install -r requirements.txt
```

Dependensi belum dikunci ulang. Versi lingkungan eksperimen terdahulu tercatat di `experiment_manifest.json` masing-masing hasil. `requirements.txt` adalah daftar dependensi, bukan jaminan reproduksi numerik lintas versi.

Dataset WAV lengkap tidak disertakan. Audio di `results/**/audio_examples/` hanya contoh keluaran. Model WavLM, HuBERT, SQUIM, dan model ASR/WER diunduh saat dibutuhkan. Eksperimen 1.000 iterasi memerlukan sumber daya besar; mode `factorial` mengikuti notebook asal yang mensyaratkan GPU CUDA.

- LibriTTS: susunan `<root>/<speaker>/<chapter>/<utterance>.wav`.
- VCTK: susunan `<root>/wav48*/pXXX/<utterance>.wav`. Resolusi audio VCTK mempertahankan preferensi `_mic1` bila ada pasangan mikrofon.
- Seleksi: 20 speaker kalibrasi × 5 utterance = 100; 40 speaker uji × (2 enrollment + 8 probe) = 400; total 500 utterance dari 60 speaker. Speaker kalibrasi dan uji terpisah.
- Durasi audio yang dipilih 1–4 detik, tanpa pemotongan/padding; normalisasi RMS target −26 dBFS dengan batas puncak 0,95.

Jalankan perintah berikut dari folder proyek. Pola lokasi WAV harus dibungkus tanda kutip.

## Cara menjalankan

Periksa konfigurasi tanpa dependensi ML, unduhan model, atau penghitungan PGD:

```powershell
python -m voice_deff --dataset libritts --mode full --dataset-glob "datasets/libri-tts-train-clean-360-part-2/**/*.wav" --dry-run
python -m voice_deff --dataset vctk --mode factorial --dataset-glob "datasets/VCTK-Corpus/wav48*/**/*.wav" --dry-run
```

`--dry-run` menghitung file yang cocok dan menampilkan rencana. Nilai nol berarti data belum ditemukan; pemeriksaan kecukupan speaker/durasi dilakukan saat eksperimen nyata. Folder output harus baru atau kosong, sehingga hasil sebelumnya tidak tertimpa.

**Mode `full`:** menghitung ulang enam metode utama dengan protokol skrip terbaru, yaitu ambang ASV tetap 0,25. Hasil default masuk ke `runs/<dataset>/full/`.

```powershell
python -m voice_deff --dataset libritts --mode full --dataset-glob "datasets/libri-tts-train-clean-360-part-2/**/*.wav"
python -m voice_deff --dataset vctk --mode full --dataset-glob "datasets/VCTK-Corpus/wav48*/**/*.wav"
```

Enam metode: Undefended, Random Noise, Static PGD, Dynamic PGD (No U-Shape), High-Frequency PGD (3.75–8 kHz), dan Proposed (Dynamic + U-Shape). Empat metode PGD menjalankan total 3.840.000 iterasi per dataset.

**Mode `factorial`:** memakai ulang manifest seleksi, seed, dan ambang dari `results/<dataset>/calibrated_asv/`, kemudian menghitung metode ketujuh, Static PGD + U-Shape. Mode ini menjalankan 960.000 iterasi tambahan, menggabungkan 6.720 baris, lalu membuat analisis empat kombinasi routing/filtering.

```powershell
python -m voice_deff --dataset libritts --mode factorial --dataset-glob "datasets/libri-tts-train-clean-360-part-2/**/*.wav"
python -m voice_deff --dataset vctk --mode factorial --dataset-glob "datasets/VCTK-Corpus/wav48*/**/*.wav"
```

Opsi tambahan: `--old-results "folder/hasil-terkalibrasi"`, `--output "runs/percobaan-baru"`, dan `--skip-wer`. Jika WER dilewati, WER hasil baru tidak dapat dibandingkan dengan hasil lama. CLI menerima sumber hasil lama berupa folder, dan menolak versi ambang tetap untuk mode `factorial` agar metrik keberhasilannya konsisten. Hasil `full` dengan ambang tetap bukan masukan langsung untuk mode `factorial` ini.

Static U-shape mempertahankan bobot State-A dan filter State-A sepanjang iterasi: frekuensi ≥2,5 kHz dikalikan 0,05. Implementasi asal ini berbeda dari penerapan kurva State-B sepanjang waktu. Nama metode dipertahankan sesuai notebook.

Keluaran utama mode `factorial`: `static_ushape_per_sample_metrics.csv`, `combined_per_sample_metrics.csv`, `factorial_speaker_summary.csv`, `factorial_pairwise_tests.csv`, `factorial_interaction_tests.csv`, `factorial_results_table.csv`, `factorial_ablation.png`, manifest dan `validation.json`. Arsip ZIP ditulis di sebelah folder output, dengan akhiran `_archive.zip`.

Empat sumber asli tetap ada di `archive/original_sources/` sebagai referensi. Skrip `.py` asli mengandung magic Jupyter (`%%capture`, `%pip`) sehingga tidak valid untuk eksekusi Python biasa. Kode aktif sudah menghapus magic dan ketergantungan lokasi `/kaggle/working`; jangan menjalankan skrip arsip sebagai pengganti CLI. Notebook arsip masih memakai lokasi sumber lama sehingga perlu penyesuaian bila dijalankan sendiri.

## Kesimpulan hasil yang tersedia

Tabel berikut mengambil nilai rata-rata dari `calibrated_asv/summary_statistics.csv`. ASV lebih rendah berarti kemiripan identitas lebih rendah; PESQ dan STOI lebih tinggi berarti kualitas/inteligibilitas lebih baik. Privacy–utility merupakan persentase keberhasilan gabungan sesuai ambang kalibrasi masing-masing dataset.

| Dataset | Metode | ASV ↓ | PESQ ↑ | STOI ↑ | Privacy–utility (%) ↑ |
|---|---|---:|---:|---:|---:|
| LibriTTS | Static PGD | 0,2520 | 1,0497 | 0,7491 | 0,0000 |
| LibriTTS | Dynamic PGD tanpa U-Shape | 0,2756 | 1,0460 | 0,7301 | 0,0000 |
| LibriTTS | High-Frequency PGD | 0,3600 | 1,3527 | 0,9612 | 10,9375 |
| LibriTTS | Proposed | 0,3167 | 1,2105 | 0,8983 | 0,1042 |
| VCTK | Static PGD | 0,2558 | 1,1461 | 0,5734 | 0,0000 |
| VCTK | Dynamic PGD tanpa U-Shape | 0,2793 | 1,1475 | 0,5616 | 0,0000 |
| VCTK | High-Frequency PGD | 0,3534 | 1,4139 | 0,7619 | 6,1458 |
| VCTK | Proposed | 0,3051 | 1,3472 | 0,6519 | 0,3125 |

Pada rata-rata yang tersedia, Proposed mempertahankan PESQ/STOI lebih baik daripada Static PGD dan Dynamic PGD tanpa filter, tetapi ASV-nya lebih tinggi. High-Frequency PGD memiliki PESQ/STOI dan keberhasilan privacy–utility lebih tinggi daripada Proposed. Jadi data ini menunjukkan pertukaran proteksi identitas dan kualitas; belum mendukung klaim Proposed unggul di semua metrik. Perbandingan ini deskriptif; keputusan signifikansi memerlukan tabel uji berpasangan sesuai protokol.

Ambang STOI cukup ketat (sekitar 0,9874 untuk LibriTTS dan 0,9710 untuk VCTK). Keberhasilan gabungan Proposed tetap rendah meskipun skor ASV turun. Undefended juga memiliki keberhasilan gabungan 6,5625% di kedua dataset, sehingga angka keberhasilan perlu dibaca bersama kondisi tanpa proteksi. SQUIM-MOS adalah nama metrik yang benar dalam kode; hasil ini bukan pengukuran NISQA.

## Duplikat dan jejak perubahan

- `reports/original_inventory.csv`: 144 lokasi awal, ukuran, dan SHA-256.
- `reports/migration_map.json`: pemetaan setiap file awal ke lokasi sekarang beserta hash.
- `reports/duplicate_files.json`: 47 alias identik dan file kanonisnya.
- `reports/code_consolidation.json`: fungsi/kelas yang disatukan dan bagian khusus protokol.
- `reports/results_comparison.csv`: ringkasan metrik kedua versi hasil.
- `reports/validation.json`: hasil pemeriksaan integritas dan kode.

Nama file tetap dipertahankan ketika berperan dalam alur eksperimen, termasuk checkpoint, preview, dan contoh audio yang kebetulan identik. Hard link membuatnya memakai satu isi fisik. Perlakukan `results/` sebagai arsip: mengedit isi satu hard link juga mengubah alias lain dalam kelompok tersebut; untuk analisis baru, tulis file baru di `runs/`. Penyalinan ke sistem berkas lain dapat menjadikannya salinan biasa lagi tanpa mengubah isi.

## Validasi dan batas pemeriksaan

```powershell
python tools/audit_workspace.py
python -m unittest discover -s tests -v
```

Audit memeriksa hash seluruh file awal, identitas hard link, kesetaraan AST 43 fungsi/kelas skrip utama dan 35 fungsi/kelas notebook, sintaks modul, keunikan pasangan sample–seed–method, kesamaan probe antar-metode, serta pemisahan speaker. Empat folder hasil berisi total 23.040 baris metrik dan lolos pemeriksaan tersebut.

Eksperimen GPU, unduhan model, dan penghitungan ulang metrik audio **belum dijalankan**. Lingkungan lokal saat pemeriksaan belum memiliki torchaudio, librosa, transformers, PESQ, dan pystoi. Kesetaraan struktur kode dan pemeriksaan data tidak menjamin kompatibilitas runtime semua versi dependensi. Skrip `tools/organize_workspace.py` adalah catatan migrasi satu kali dan menolak dijalankan ulang setelah migrasi selesai.
