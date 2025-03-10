# Pertemuan 1: Pengantar *Machine Learning* pada bidang Teknik Elektro

**[TKE1686] MK Machine Learning (3-SKS)**

oleh Gramandha Wega Intyanto


## 1. Definisi Machine Learning
Machine Learning (ML) adalah cabang dari kecerdasan buatan (Artificial Intelligence, AI) yang memungkinkan sistem untuk belajar dari data tanpa diprogram secara eksplisit. ML menggunakan algoritma dan model statistik untuk mengenali pola dalam data dan membuat keputusan atau prediksi berdasarkan pola tersebut.

### Definisi dari beberapa ahli
> **Arthur Samuel (1959)**: "*Machine Learning* adalah bidang studi terkait pemberian kemampuan pada komputer untuk belajar tanpa diprogram secara eksplisit."

> **Tom Mitchell (1997)**: 
"*Machine Learning* adalah suatu program komputer yang belajar dari pengalaman *(experience E)* terhadap tugas *(Task T)* dengan ukuran performa *(performance P)*, jika performanya terhadap tugas *(Task T)* meningkat dengan pengalaman *(experience E)*."

Dapat disimpulkan bahwa **machine learning** merupakan bidang ilmu yang mempelajari dari data dan mengeksekusi hasil dari pembelajaran data tersebut.


## 2. Sejarah Singkat Machine Learning
### 1950-an:
- Alan Turing memperkenalkan Turing Test untuk mengukur kecerdasan mesin.
- Frank Rosenblatt mengembangkan perceptron, model awal dari neural network.
### 1970-an s.d. 1980-an:
- Pengembangan algoritma decision tree dan neural network pertama.
- Backpropagation mulai digunakan dalam pelatihan neural network.
### 1990-an:
- Munculnya Support Vector Machine (SVM) dan Random Forest sebagai metode klasifikasi canggih.
- Mulai berkembangnya aplikasi ML dalam pattern recognition dan data mining.
### 2000-an s.d. Sekarang:
- Kemajuan dalam Deep Learning dan Neural Networks berkat peningkatan daya komputasi dan ketersediaan big data.
- Model seperti Convolutional Neural Network (CNN) dan Recurrent Neural Network (RNN) menjadi dasar dalam pengolahan citra dan sinyal.
- Aplikasi ML berkembang pesat di berbagai bidang, termasuk kesehatan, finansial, transportasi, dan teknik elektro.

## 3. Klasifikasi Machine Learning yang akan di pelajari

Ditarik dari permasalahan pembelajaran _(learning problem)_ dalam konteks ML pada dataset (sejumlah _n_ sample data) untuk melakukan prediksi terhadap properties yang tidak diketahui pada dataset lain yang sejenis. 
Secara umum untuk penyelesain permasalahan pembelajaran, _Machine Learning_ dibagi menjadi dua teknik dasar yang akan kita pelajari pada mata kuliah (2), yaitu **supervised learning** dan **unsupervised learning**

### Supervised Learning

Teknik atau model pembelajaran yang diterapkan pada mesin dari pola dalam data yang memiliki label tertentu

contoh kasus: 
1. Deteksi kesalah paa jaringan listrik, dimana data yang dimiliki yaitu tegangan, arus, frekuensi dalam jaringan. Kemudian tiap data memiliki label jaringan berstatus: **Normal** dan **Gangguan**
2. Robot akan melakukan klasifikasi jenis bunga, dimana data yang memiliki kelopak bunga, bentuk kelopak, warna, dsb. Kemudian tiap data memiliki label jenis bunga:  **Mawar** dan **Bunga Matahari**

Pada supervised learning ini pada umumnya memiliki minimal 2 data, data input yaitu pola pada data dan data output yaitu kategori (label).

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20241022160725494723/supervised-machine-learning.webp" width="500">
source : geeksforgeeks


Tipe dari _Supervised Learning_ yaitu:

1. Klasifikasi: Di mana output adalah variabel kategoris (mis., Email spam vs non-spam, ya vs tidak). 
2. Regresi: Di mana output adalah variabel kontinu (mis., Memprediksi harga rumah, harga saham).

Contoh algoritma atau model yang sering digunakan pada bidang elektro yaitu Regresi, Suppert Vector Machine (SVM), K-Nearest Neighbors (K-NN), Random Forest biasanya digunakan untuk klasifikasi, dsb.

Berikut contoh algoritma yang lebih lengkap terkait _Supervised Learning_

| Algoritma         | Regresi, Klasifikasi      | Tujuan                                          | Metode                                                      | Kasus Penggunaan                                      |
|-------------------|--------------------------|------------------------------------------------|-------------------------------------------------------------|------------------------------------------------------|
| **Regresi Linear**    | Regresi                   | Memprediksi nilai output kontinu                | Persamaan linear yang meminimalkan jumlah kuadrat residu    | Memprediksi nilai kontinu                           |
| Regresi Logistik  | Klasifikasi               | Memprediksi variabel output biner               | Fungsi logistik yang mentransformasikan hubungan linear     | Tugas klasifikasi biner                            |
| Pohon Keputusan   | Keduanya                  | Memodelkan keputusan dan hasil                  | Struktur pohon dengan keputusan dan hasil                   | Tugas klasifikasi dan regresi                      |
| Random Forest     | Keduanya                  | Meningkatkan akurasi klasifikasi dan regresi    | Menggabungkan beberapa pohon keputusan                     | Mengurangi overfitting, meningkatkan akurasi prediksi |
| **SVM**               | Keduanya                  | Membuat hyperplane untuk klasifikasi atau prediksi nilai kontinu | Memaksimalkan margin antara kelas atau memprediksi nilai kontinu | Tugas klasifikasi dan regresi  |
| **KNN**               | Keduanya                  | Memprediksi kelas atau nilai berdasarkan k tetangga terdekat | Mencari k tetangga terdekat dan memprediksi berdasarkan mayoritas atau rata-rata | Tugas klasifikasi dan regresi, sensitif terhadap data berisik |
| Gradient Boosting | Keduanya                  | Menggabungkan model lemah untuk membuat model yang kuat | Mengoreksi kesalahan secara iteratif dengan model baru     | Tugas klasifikasi dan regresi untuk meningkatkan akurasi prediksi |
| Naive Bayes       | Klasifikasi               | Memprediksi kelas berdasarkan asumsi independensi fitur | Teorema Bayes dengan asumsi independensi fitur              | Klasifikasi teks, penyaringan spam, analisis sentimen, medis |


### Unsupervised Learning

Teknik atau model pembelajaran yang diterapkan pada mesin dari pola dalam data yang tidak memiliki label tertentu

Contoh kasus

Operator jaringan lisrik ingin mengkelompokkan pelanggan berdasarkan pola komsumsi energi untuk meningkatan efiseiensi ditribusi daya. 

Pada _unsupervised learning_ ini pada umumnya memiliki minimal 1 data, data input yaitu pola pada data.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20231124111325/Unsupervised-learning.png" width="500">
source : geeksforgeeks

Tipe dari _Unsupervised Learning_ yaitu:

1. **Clustering Algorithms** : Pengelompokan dalam pembelajaran mesin tanpa pengawasan adalah proses pengelompokan data yang tidak berlabel ke dalam kelompok berdasarkan kesamaan mereka. Tujuan pengelompokan adalah untuk mengidentifikasi pola dan hubungan dalam data tanpa pengetahuan sebelumnya tentang makna data. 
2. Association Rule Learning : Pembelajaran aturan asosiasi juga dikenal sebagai penambangan aturan asosiasi adalah teknik umum yang digunakan untuk menemukan asosiasi dalam pembelajaran mesin tanpa pengawasan.
3. Dimensionality Reduction : Pengurangan dimensi adalah proses mengurangi jumlah fitur dalam dataset sambil menjaga informasi sebanyak mungkin. Teknik ini berguna untuk meningkatkan kinerja algoritma pembelajaran mesin dan untuk visualisasi data.

Contoh algoritma atau model yang digunakan pada yaitu K-means cluster

#### Contoh dataset ML

Supervised

<img src="https://ilmudatapy.com/wp-content/uploads/2020/07/klasifikasi-1.png" width="250">

Unsupervised

<img src="dataset_image.png" width="250">




## 3. Peran Machine Learning dalam Teknik Elektro

Machine Learning memiliki peran penting dalam berbagai aspek Teknik Elektro, termasuk pemrosesan sinyal, pengendalian sistem, dan optimasi kinerja perangkat elektronik. Beberapa penerapan utama ML dalam Teknik Elektro adalah:
### a. Pemrosesan Sinyal dan Citra Digital
- Deteksi dan klasifikasi sinyal menggunakan Fourier Transform dan Wavelet Transform.
- Pengenalan pola dalam sinyal kelistrikan, seperti fault detection dalam jaringan listrik.
- Penerapan ML dalam pengolahan citra digital untuk mendeteksi objek atau menganalisis data dari kamera industri.
### b. Sistem Kendali dan Otomasi
- Machine Learning digunakan dalam sistem kontrol adaptif, misalnya pada robotik dan otomasi industri.
- Peningkatan efisiensi sistem kendali berbasis fuzzy logic dan neural network.
### c. Prediksi dan Optimasi dalam Jaringan Listrik
- Memprediksi konsumsi daya listrik menggunakan time-series forecasting.
- Optimasi distribusi daya menggunakan ML untuk meningkatkan efisiensi grid listrik.
### d. Machine Learning dalam Embedded Systems
- Implementasi TinyML (Machine Learning pada perangkat dengan daya rendah seperti mikroprosesor dan FPGA).
- Aplikasi ML dalam Internet of Things (IoT) untuk sistem pemantauan cerdas.
### e. Keamanan dan Deteksi Anomali
- Pendeteksian anomali dalam jaringan listrik untuk mencegah blackout atau gangguan listrik.
- Sistem keamanan berbasis ML dalam deteksi peretasan jaringan komunikasi dan kendali.



## 4. Tools yang digunakan untuk ML
Pada akhir tahun ini perancangan ML bahasa program yang sering digunakan yaitu Python dengan beberapa tools library berikut:

Framework dan Library ML
- [TensorFlow](https://www.tensorflow.org/) : Framework open-source untuk deep learning dan ML yang dikembangkan oleh Google.
- [PyTorch](https://pytorch.org/)  :   Framework ML berbasis Python dengan fleksibilitas tinggi, dikembangkan oleh Facebook AI.
- [Scikit-learn](https://scikit-learn.org/) : Library ML berbasis Python untuk klasifikasi, regresi, dan clustering.
- [Keras](https://keras.io/)  :   API tinggi berbasis TensorFlow untuk pengembangan model deep learning.

Tools untuk Pemrosesan Data dan Visualisasi
- [Pandas](https://pandas.pydata.org/) :   Library Python untuk manipulasi dan analisis data berbasis tabel.
- [NumPy](https://numpy.org/) :   Library untuk operasi numerik dan array multidimensi dalam Python.
- [Matplotlib](https://matplotlib.org/) : Library visualisasi untuk membuat grafik dan plot data.
- [Seaborn](https://seaborn.pydata.org/) : Library berbasis Matplotlib untuk visualisasi data yang lebih menarik.
- [Dask](https://www.dask.org/) : Library untuk pemrosesan data besar yang dapat berjalan secara paralel.

*Note: Kalau Anda ingin mencoba-coba ada software yang sudah include untuk penerapan ML yaitu [WEKA](https://ml.cms.waikato.ac.nz/weka/index.html)


## 4. Tugas Mahasiswa

1. Silahkan cari studi kasus pada bidang Anda yang dapat diselesaikan dengan ML. Beri 2 contoh tiap teknik(**_supervised learning_** dan **_unspervised learning_**) satu masalah!
2. Cari dataset terkait studi kasus tersebut kemudian jelaskan proses penyelesaian dengan algoritma apa!

**Note:**
Rekomendasi tempat untuk mencari dataset: [kaggle](https://www.kaggle.com/datasets), [uci](https://archive.ics.uci.edu/), [google dataset](https://datasetsearch.research.google.com/)
