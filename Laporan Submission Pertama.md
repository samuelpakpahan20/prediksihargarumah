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

### Univariate Analysis

-   Distribusi `bedrooms`

    ![Distribusi Bedrooms](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/bedrooms.JPG)
    - Dapat dilihat bahwa sebagian besar jumlah kamar tidur itu di angka 3 dan 4.
    - Data memiliki banyak outliers.


-   Distribusi `bathrooms`

    ![Distribusi Bathrooms](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/bathrooms.JPG)
    - Jumlah kamar mandi paling banyak berada pada angka 1 dan 2.
    - Kemudian, terdapat rumah yang tidak ada kamar mandinya atau jumlahnya 0
    - Nilai outliernya lumayan banyak.


-   Distribusi `sqft_living`

    ![Distribusi Living Room](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/livingroom.JPG)
    - Density dari distribusi luas rumah berada di sekitar angka 2000an.
    - Banyak terdapat outliers.


-   Distribusi `grade`

    ![Distribusi Grade](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/grade.JPG)
    - Sebagian besar rumah di County King US memiliki grade 7 dan 8.
    - Dilihat dari boxplot, data memiliki beberapa outliers.


-   Distribusi `yr_built`

    ![Distribusi Tahun](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/year.JPG)
    - Dapat dilihat bahwa semakin tua umur dari rumah, maka semakin sedikit orang yang menjual rumahnya tersebut.
    - Density terdapat di sekitar tahun 1980an.
    - Data tidak memiliki outliers.

### Multivariate Analysis

-   Hubungan antara variabel independent dan variabel dependent

    Perlu diingat : 
    - Independent variabel (x) adalah `bedrooms`, `bathrooms`, `sqft_living`, `grade`, dan `yr_built`.
    - Dependent variabel (y) adalah `price`.
    
    ![Plot variabel](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/multivariate.JPG)
    
    Nilai korelasinya dapat dilihat pada tabel berikut.
    
    *ps : Hasil ini tidak akan selalu sama*.
    
    ![Tabel korelasi](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/korelasi.JPG)
    - Dari tabel korelasi diatas, dapat dilihat bahwa `sqft_living` mempunyai hubungan linear positif yang sangat kuat dengan `price` jika dibandingkan yang lain.
    - Nilai korelasi `yr_built` hampir mendekati nol yang menandakan bahwa usia rumah tidak mempengaruhi pada harga rumah.

## Evaluation

Seperti yang sudah dijelaskan pada tab [Solution statements](#solution-statements), pada proyek ini model yang dibuat merupakan kasus prediksi dan menggunakan algoritma **Multiple Linear Regression**. Pada gambar dibawah ini ditampilkan kembali formula dari Multiple Linear Regression / Regresi Linear Berganda.

![Rumus Multiple Linear Regression](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/rumusmlr.JPG)

*Keterangan:*
- Y = dependent variable
- mn = koefisien dari persamaan
- xn = independent variable
- b = intercept
- e = error

Setelah kita melakukan data preparation dengan mendapatkan fitur-fitur terbaik serta melihat korelasi antar fitur variable dan fitur target. Selanjutnya, kita akan membuat model menggunakan Algoritma Multiple Linear Regression. Untuk menggunakan algoritma tersebut, masukkan kode berikut:
```
# Pertama, buat variabel x dan y
x = df_features.drop(columns='price')
y = df_features['price']

# Kedua, kita split data kita menjadi 80% training dan 20% testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Ketiga, kita bikin object linear regresi
lin_reg = LinearRegression()

# Keempat, train the model menggunakan training data yang sudah displit
lin_reg.fit(x_train, y_train)

# Kelima, cari tau nilai slope/koefisien (m) dan intercept (b)
# Kita coba buat kedalam dataframe agar kebih rapi
coef_dict = {
    'features': x.columns,
    'coef_value':lin_reg.coef_
}
coef = pd.DataFrame(coef_dict, columns=['features', 'coef_value'])
coef
```

Hasilnya sebagai berikut.

![Dataframe MLR](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/dataframe.JPG)

Dari nilai m dan b diatas, kalau dimasukan ke dalam rumus menjadi:

Y = -49110.86x1 + 62897.89x2 + 183.65x3 + 131451.54x4 - 4075.54x5 + 7217062

Kemudian, kita cari tahu accuracy score dari model kita menggunakan testing data yang sudah displit. Gunakan kode berikut.
```
lin_reg.score(x_test, y_test)
```

Hasilnya seperti berikut.

![Skor Akurasi](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/akurasi.JPG)

Model kita mendapatkan accuracy score sebesar 61.01%

Terakhir mari kita prediksi harga rumah sesuai dengan permintaan seseorang, sebut saja namanya Samuel dengan kriteria sebagai berikut:
- bedrooms = 3
- bathrooms = 2
- sqft_living = 1800 sqft
- yr_built = 1990

Masukkan kode berikut.
```
#Prediksi harga rumah idaman Samuel
lin_reg.predict([[3,2,1800,7,1990]])
```

Hasilnya,

![Harga Prediksi Rumah](https://raw.githubusercontent.com/samuelpakpahan20/prediksihargarumah/master/images/harga.JPG)

Maka Harga rumah idaman Samuel adalah sekitar US$ 335936

*ps : Semua hasil dari pemodelan ini tidak akan selalu sama untuk tiap kali dijalankan*.

**---Ini adalah bagian akhir laporan---**
