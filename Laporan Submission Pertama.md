# Laporan Proyek Machine Learning - Samuel Partogi Pakpahan

## Domain Proyek

-   Latar Belakang

<p align="center">
  <img width="460" height="300" src="https://static.seattletimes.com/wp-content/uploads/2020/06/06052020_homesales_104503-1560x999.jpg" alt="Sumber : https://www.seattletimes.com/business/real-estate/king-county-home-prices-dropped-as-coronavirus-squelched-activity-but-now-the-market-may-be-picking-up/">
</p>

Perumahan dan pemukiman merupakan salah satu kebutuhan pokok manusia. Selain itu rumah juga merupakan kebutuhan dasar manusia dalam meningkatkan harkat, martabat, mutu kehidupan dan penghidupan. Serta sebagai pencerminan diri pribadi dalam upaya peningkatan taraf hidup, serta pembentukan watak, karakter dan kepribadian bangsa. Rumah juga menjadi salah satu bentuk investasi yang menarik. Seperti saat ini, kebutuhan perumahan di wilayah King County mengalami penurunan sejak adanya Virus Corona. Program _stay at home_ membuat banyak pembeli dan penjual potensial keluar dari pasar pada bulan April 2020, contohnya saja pada awal pandemi virus corona harga di King County jatuh pada bulan Mei berturut-turut. Harga rata-rata turun 4% dari tahun ke tahun, menjadi $672.000 [[1](https://www.seattletimes.com/business/real-estate/king-county-home-prices-dropped-as-coronavirus-squelched-activity-but-now-the-market-may-be-picking-up/)]. Harga di King County juga turun 6% dari April hingga Mei, pertama kalinya sejak 2014 nilai rumah di King County jatuh antara dua bulan yang biasanya membawa hiruk-pikuk pembelian rumah di musim semi. Laporan dari [Northwest Multiple Listing Service](https://www.seattlepi.com/realestate/article/Here-s-how-the-2020-housing-market-in-King-County-15884787.php), ditemukan pada tahun 2020 di King County, ada 37.824 daftar penjualan rumah yang tertunda, penurunan hampir 32% dari tahun ke tahun dibandingkan pada tahun 2019 yang memiliki 55.218 daftar penjualan rumah yang tertunda. Dan pada tahun 2020, King County memiliki 25.373 daftar rumah hunian aktif, penurunan 40% dari 2019.

Dengan adanya masalah penurunan harga serta penundaan penjualan rumah ini, maka pemilhan dana rumah yang bijak harus dimiliki. Sehingga, pada proyek ini perlu dibuat sebuah model _machine learning_ untuk memprekdisi harga rumah di wilayah King County, USA. Dengan adanya model _machine learning_ ini diharapkan masyarakat King County dapat dengan mudah untuk mencari perumahan yang diinginkan dengan harga yang sesuai. 

## Business Understanding

### Problem Statements

Berangkat dari latar belakang diatas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini :

-   Bagaimana cara membuat model _machine learning_ untuk memprediksi harga rumah?
-   Bagaimana cara menentukan prediksi harga?

### Goals

Berikut adalah tujuan dari dibuatnya proyek ini :

-   Membuat model _machine learning_ untuk memprediksi harga rumah.
-   Memprediksi harga rumah sesuai dengan kriteria pembeli.

### Solution statements

Untuk pembuatan model dipilih penggunaan model dengan algoritma **Multiple Linear Regression**. Algoritma tersebut dipilih karena mudah cocok untuk kasus ini dan sesuai dengan datasetnya. Algoritma ini digunakan untuk mengetahui pengaruh beberapa variabel bebas (independent) terhadap variabel terikat (dependent). Cara menghitungnya adalah dengan rumus berikut ini :

![Rumus Multiple Linear Regression](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/rumusmlr.JPG)

*Keterangan:*
- Y = dependent variable
- mn = koefisien dari persamaan
- xn = independent variable
- b = intercept
- e = error

Selain itu, berikut ini merupakan kelebihan dan kekurangan dari algoritma Multiple Linear Regression:
- Kelebihan :
  - Mudah dipahami dan mudah digunakan
  - Sangat kompleks dibandingkan dengan algoritma lain
  - Dapat Memprediksi Tren di Masa yang Akan Datang
- Kekurangan yang paling mencolok adalah karena hasil ramalan dari analisis regresi merupakan nilai estimasi, sehingga kemungkinan untuk tidak sesuai dengan data aktual tetaplah ada. Selain itu, penentuan variabel independen dan variabel dependen yang saling berkaitan dalam hal sebab-akibat juga terbilang cukup susah, karena bisa jadi model yang tidak cukup bagus disebabkan karena kesalahan dalam memilih variabel yang digunakan untuk analisis. Misalkan data gaji pegawai tidak berkaitan dengan tempat anaknya bersekolah, sehingga jika menggunakan variabel tersebut model yang didapatkan tidak akan bagus.


## Data Understanding

![Sampul Dataset](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/sampul.JPG)

Informasi dataset dapat dilihat pada tabel dibawah ini :

| Jenis                   | Keterangan                                                                                                |
| ----------------------- | --------------------------------------------------------------------------------------------------------- |
| Sumber                  | [Kaggle Dataset : House Sales in King County, USA](https://www.kaggle.com/harlfoxem/housesalesprediction) |
| Lisensi                 | CC0: Public Domain                                                                                        |
| Kategori                | Keuangan                                                                                                  |
| Rating Penggunaan       | 7.1                                                                                                       |
| Jenis dan Ukuran Berkas | CSV (3 MB)                                                                                                |

Pada berkas yang diunduh yakni `kc_house_data.csv` berisi informasi data historis rumah yang terjual antara Mei 2014 hingga Mei 2015 di King County, dan juga mencakup Seattle. Dataset terdiri dari 21 variabel dan 21.613 sampel. Untuk penjelasan mengenai variabel-variable pada data _kc house data_ dapat dilihat pada poin-poin berikut :

1. `id` : kode unik untuk setiap rumah yang dijual.
2. `date` : tanggal dimana rumah tersebut terjual.
3. `price`: harga rumah yang harus di prediksi jadi ini adalah variabel target.
4. `bedrooms` : jumlah kamar tidur di sebuah rumah.
5. `bathrooms` : jumlah kamar mandi di kamar tidur/sebuah rumah.
6. `sqft_living` : ukuran rumah dalam kaki persegi.
7. `sqft_lot` : pengukuran yang menentukan kaki persegi lot.
8. `floors`: total lantai/tingkat rumah.
9. `waterfront` : fitur ini menentukan apakah sebuah rumah memiliki view ke waterfront (0 berarti tidak, 1 berarti ya).
10. `view` : fitur ini menentukan apakah suatu rumah sudah dilihat atau belum (0 berarti tidak 1 berarti ya).
11. `condition` :kondisi keseluruhan rumah pada skala 1 sampai 5.
12. `grade` : nilai keseluruhan yang diberikan ke unit perumahan, berdasarkan sistem penilaian King County pada skala 1 sampai 11
13. `sqft_above` : ukuran luas persegi rumah selain basement.
14. `sqft_basement` : luas persegi ruang bawah tanah rumah.
15. `yr_built` : tanggal pembangunan rumah.
16. `yr_renovated` : tahun renovasi rumah.
17. `zipcode` : kode pos lokasi rumah.
18. `lat` : garis lintang letak rumah.
19. `long` : garis bujur letak rumah.
20. `sqft_living15` : Ukuran area ruang tamu pada tahun 2015
21. `sqft_lot15` : ukuran area lot pada tahun 2015

## Data Preparation

Berikut adalah tahapan-tahapan dalam melakukan pra-pemrosesan data :

- Dari hasil pengamatan data, dataset ini memiliki banyak kolom tambahan tentang sebuah rumah dan ada beberapa kolom yang kurang berkolerasi dengan kolom `harga`. Jadi, kita hanya pakai kolom `bedrooms`, `bathrooms`, `sqft_living`, `grade`, `price`, dan `yr_built` untuk digunakan sebagai kumpulan fitur terbaiknya.

- Merubah tipe data. 
  Dari hasil statistical description, dapat diketahui bahwa pada kolom `bathrooms` memiliki nilai pecahan, hal ini sangat aneh jika suatu rumah memiliki jumlah kamar mandi pecahan. Maka tipe datanya diubah dari `float` menjadi `int`.

- Memodifikasi nilai. 
  Pada kolom `bedrooms` terdapat nilai 33, hal ini sangat aneh karena sebuah rumah pribadi tidak mungkin ada yang mempunyai jumlah kamar 33. Hal ini kemungkinan terjadi karena kesalahan pada saat menginput nilai (typo), jadi kita akan ganti menjadi 3.

- Mengecek Missing Value. 
  Namun dataset ini tidak memiliki missing value.

## Modeling

Setelah melakukan pra-pemrosesan data yang baik, selanjutnya kita akan melakukan **Exploratory Data Analysis** **_(EDA)_** dengan dua cara, yakni Univariate dan Multivariate Analysis.

-   Model _baseline_

    Pada tahap ini saya membuat model dasar dengan menggunakan _modul_ scikit-learn yakni [KNeighborsClassifier](https://scikit-learn.org/0.24/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier) tanpa menggunakan parameter tambahan. Lalu melakukan prediksi kepada data ujinya.

-   Model yang dikembangkan

    Kemudian setelah melihat kinerja model _baseline_, agar dapat bekerja lebih optimal lagi maka digunakan sebuah fungsi untuk mencari _hyperparameter_ yang optimal dengan [HalvingGridSearchCV](https://scikit-learn.org/0.24/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html#sklearn.model_selection.HalvingGridSearchCV). Setelah ditemukan yang optimal, kemudian _hyperparameter_ tersebut diterapkan ke model _baseline_.

Hasilnya dapat dilihat seperti pada tabel berikut ini :

![performa model](https://user-images.githubusercontent.com/58651943/133832133-83717305-7a38-4c33-b206-91fa519d0e19.png)

Pada model _baseline_ nilai akurasinya cukup buruk. Begitupun nilai _f1-score_, _recall_ dan _precision_ pada setiap labelnya. Namun setelah dilakukan pengaturan _hyperparameter_, nilai akurasi pun meningkat. Begitupun nilai _f1-score_, _recall_ dan _precision_ pada setiap labelnya. Untuk membuktikannya, kedua model tersebut diuji pada data uji dan di visualisasikan pada _confussion matrix_ seperti berikut.

-   Model _baseline_

![performa model baseline](https://user-images.githubusercontent.com/58651943/133832795-a6cc120e-1153-42d3-967f-a0a22df4c9e3.png)

-   Model yang dikembangkan

![performa model improvement](https://user-images.githubusercontent.com/58651943/133833185-32e102f2-6d81-4f00-b06b-b8f3c7313903.png)

Dengan hasil diatas, maka model yang dikembangkan merupakan model yang dipilih untuk digunakan.

## Evaluation

Pada proyek ini, model yang dibuat merupakan kasus klasifikasi dan menggunakan metriks akurasi, _f1-score_, _recall_ dan _precision_. Pada gambar dibawah ini ditampilkan kembali hasil pengukuran model yang dikembangkan dengan metriks akurasi, _f1-score_, _recall_ dan _precision_.

![performa model improvement eval](https://user-images.githubusercontent.com/58651943/133834417-3d8e57b8-9546-4dc9-b5b7-7f58a1abc4ea.png)

-   Akurasi

![formula akurasi sklearn](https://user-images.githubusercontent.com/58651943/133834677-91c885d0-a443-4567-b75b-30f106ac8124.png)

Akurasi merupakan metrik untuk menghitung nilai ketepatan model dalam memprediksi data dengan data yang sebenarnya. Akurasi dapat dihitung dengan rumus diatas. Kelebihan dari metriks ini adalah sering digunakan dalam kasus pembuatan model klasifikasi baik itu klasifikasi dua kelas, atau kategori. Kekurangan dari metrik ini adalah dapat bersifat 'menyesatkan' pada data yang tidak seimbang.

-   _precision_

_Precision_ merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik model memprediksi label positif terhadap semua prediksi model berlabel positif. Lalu bagaimana cara menghitungnya, pertama-tama kita perlu mengenali dulu istilah TP,TN,FP,FN. Penjelasan singkatnya dapat dilihat pada tabel dibawah ini

![tp,tn,fp,fn](https://user-images.githubusercontent.com/58651943/133837008-ce49e685-d592-475e-b6b9-00e007123a47.png)

Setelah memahaminya, kitapun dapat menghitungnya dengan rumus dibawah ini

![formula precision sklearn](https://user-images.githubusercontent.com/58651943/133837478-fe8bb36a-8964-4133-8cad-d7ad308e6bff.png)

Kelebihan dari metriks ini berfokus pada bagaimana performa (prediksi) model terhadap label data positif, kekurangannya metriks ini tidak memperhitungkan label negatifnya.

-   _Recall_

_Recall_ merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik model memprediksi label positif terhadap semua label data positif. Cara menghitungnya dapat dilihat pada rumus dibawah ini

![formula recall sklearn](https://user-images.githubusercontent.com/58651943/133840605-edcd7b7e-2b82-44fe-8fd8-7acecf754c55.png)

Kelebihan dari metriks ini menghitung bagian negatif dari prediksi label positif (tidak seperti precision). Tetapi kekurangannya ketika semua prediksi = 1 maka _recall_ akan bernilai 1 (tidak memperhitungkan prediksi negatif).

-   _f1-score_

_f1-score_ merupakan metrik dalam kasus klasifikasi yang digunakan untuk menghitung seberapa baik hasil prediksi model (_precision_) dan seberapa lengkap hasil prediksinya (_recall_). Cara menghitungnya dapat dilihat pada rumus dibawah ini

![formula f1-score sklearn](https://user-images.githubusercontent.com/58651943/133841853-6482710c-b233-4697-8bd7-2bb709e27eaf.png)

Catatan : Nilai beta = 1 (f1-score)

Kelebihan dari metriks ini menutup semua kekurangan yang ada pada _precision_ dan _recall_. Namun kekurangannya adalah _f1-score_ tidak memperhitungkan hasil prediksi benar pada label negatif.

## _Referensi:_

[[1](https://www.nature.com/articles/s41545-020-00085-z)] Bain, R., Johnston, R. & Slaymaker, T. _Drinking water quality and the SDGs._ npj Clean Water 3, 37 (2020). https://doi.org/10.1038/s41545-020-00085-z

[[2](https://www.sciencedirect.com/science/article/abs/pii/S2214714419304453)] Hasan, H. A., & Muhammad, M. H. (2020). _A review of biological drinking water treatment technologies for contaminants removal from polluted water resources._ Journal of Water Process Engineering, 33, 101035. https://doi.org/10.1016/j.jwpe.2019.101035

[[3](https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761)] Harrison, O. (2019, July 14). _Machine Learning Basics with the K-Nearest Neighbors Algorithm_. Medium. https://towardsdatascience.com/machine-learning-basics-with-the-k-nearest-neighbors-algorithm-6a6e71d01761

[[4](https://statisticsbyjim.com/basics/remove-outliers/)] Frost, J. (2021, April 5). _Guidelines for Removing and Handling Outliers in Data._ Statistics By Jim. https://statisticsbyjim.com/basics/remove-outliers/

[[5](https://towardsdatascience.com/tuning-hyperparameters-part-i-successivehalving-c6c602865619)] Descamps, B. (2018, July 3). _Tuning Hyperparameters (part I): SuccessiveHalving._ Medium. https://towardsdatascience.com/tuning-hyperparameters-part-i-successivehalving-c6c602865619

**---Ini adalah bagian akhir laporan---**
