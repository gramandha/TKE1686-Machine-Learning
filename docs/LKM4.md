# LKM 4

## **Soal Klasterisasi dengan k-NN dan Hierarchical Clustering**

### **Deskripsi Soal**
Sebuah perusahaan ingin mengelompokkan pelanggan berdasarkan pola pembelian mereka.  
Data yang tersedia mencakup informasi **penghasilan bulanan (dalam juta rupiah)** dan **jumlah transaksi dalam sebulan**.  
Gunakan **k-NN** dan **Hierarchical Clustering** untuk mengelompokkan pelanggan dan tentukan pola yang terbentuk.  

### **Data Pelanggan**

| ID Pelanggan | Penghasilan (Juta Rupiah) | Jumlah Transaksi per Bulan |
|-------------|---------------------------|----------------------------|
| P1          | 3.5                       | 5                          |
| P2          | 7.0                       | 20                         |
| P3          | 2.0                       | 3                          |
| P4          | 5.5                       | 10                         |
| P5          | 8.0                       | 25                         |
| P6          | 1.5                       | 2                          |
| P7          | 6.0                       | 12                         |
| P8          | 9.5                       | 30                         |
| P9          | 4.0                       | 7                          |
| P10         | 3.0                       | 4                          |

---

### **Tugas**
#### **1. K-NN Clustering**
- Jika kita sudah memiliki klaster awal dengan pusat klaster sebagai berikut:
  - **Klaster 1**: (2.5, 4)
  - **Klaster 2**: (7.5, 22)
- Gunakan metode **k-NN** untuk mengelompokkan pelanggan ke dalam salah satu dari klaster tersebut berdasarkan jarak Euclidean.
- Tentukan klaster akhir setelah semua pelanggan dikategorikan.

#### **2. Hierarchical Clustering**
- Gunakan metode **Agglomerative Hierarchical Clustering** dengan pendekatan **Single Linkage** atau **Complete Linkage**.
- Buat **dendrogram** untuk menunjukkan proses penggabungan klaster hingga terbentuk satu klaster besar.
- Tentukan **jumlah klaster optimal** berdasarkan dendrogram.

## **Soal Klasifikasi dengan k-NN dan Decision Tree**

### **Deskripsi Soal**
Sebuah bank ingin mengklasifikasikan calon nasabah ke dalam kategori **"Layak Kredit"** atau **"Tidak Layak Kredit"** berdasarkan beberapa parameter.  
Data yang tersedia mencakup **penghasilan bulanan**, **jumlah pinjaman yang sedang berjalan**, dan **status pekerjaan**.  
Gunakan **k-NN** dan **Decision Tree** untuk melakukan klasifikasi dan tentukan pola yang terbentuk.

---

### **Data Nasabah**

| ID Nasabah | Penghasilan (Juta Rupiah) | Pinjaman Aktif (Juta) | Status Pekerjaan  | Status Kredit (Label) |
|-----------|---------------------------|----------------------|------------------|----------------------|
| N1        | 5.0                       | 10                   | Tetap            | Layak Kredit        |
| N2        | 2.5                       | 5                    | Kontrak          | Tidak Layak Kredit  |
| N3        | 7.0                       | 20                   | Tetap            | Layak Kredit        |
| N4        | 3.0                       | 8                    | Kontrak          | Tidak Layak Kredit  |
| N5        | 6.0                       | 15                   | Tetap            | Layak Kredit        |
| N6        | 4.0                       | 12                   | Kontrak          | Tidak Layak Kredit  |
| N7        | 8.5                       | 25                   | Tetap            | Layak Kredit        |
| N8        | 3.5                       | 6                    | Kontrak          | Tidak Layak Kredit  |
| N9        | 6.5                       | 18                   | Tetap            | Layak Kredit        |
| N10       | 2.0                       | 4                    | Kontrak          | Tidak Layak Kredit  |

---

### **Tugas**
#### **1. K-NN Classification**
- Gunakan metode **k-Nearest Neighbors (k-NN)** untuk mengklasifikasikan nasabah baru dengan fitur berikut:
  - **Penghasilan** = 4.5 juta rupiah  
  - **Pinjaman Aktif** = 9 juta rupiah  
  - **Status Pekerjaan** = Kontrak  
- Gunakan **k = 3** dan hitung **jarak Euclidean** untuk menentukan kelas nasabah tersebut.

#### **2. Decision Tree Classification**
- Gunakan metode **Decision Tree** untuk membangun model klasifikasi berdasarkan fitur yang tersedia.
- Gambarkan struktur **decision tree** yang terbentuk.
- Gunakan model untuk memprediksi status kredit nasabah dengan parameter berikut:
  - **Penghasilan** = 6.0 juta rupiah  
  - **Pinjaman Aktif** = 14 juta rupiah  
  - **Status Pekerjaan** = Tetap  

---


```python

```
