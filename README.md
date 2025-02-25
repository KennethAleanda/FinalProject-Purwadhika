# FinalProject-Purwadhika
## Domain Knowledge

### Context
syarah.com adalah online marketplace untuk penjualan mobil bekas di saudi arabia, untuk menaikkan revenue akan menawarkan feature price suggestion kepada seller. feature ini akan dapat diakses oleh seller melalui subscription. dengan feature ini, seller dapat memasukkan input berupa attribute/feature dari mobil yang ingin dijual, outputnya adalah price range yang direkomendasikan untuk mobil dengan attribute/feature tersebut. 

### Stakeholder
syarah.com, syarah company (salah satu marketplace online untuk penjualan mobil bekas di Saudi Arabia)

### Busines Problem
1. **menentukan harga yang comptetive pada mobil bekas adalah hal yang lumayan sulit dilakukan**. tidak semua seller mengetahui hal-hal mengenai mobil yang ingin dijual, dan mencari informasi mengenai harga pasar dari mobil dengan feature-feature detail tertentu (e.g. mileage, kondisi kesehatan ban, kondisi exterior mobil, warna mobil tertentu, kondisi mesin, modifikasi mobil, keseringan service, tempat service dst.) sangatlah memakan waktu dan pengetahuan mengenai detail-detail tersebut. untuk kebanyakan orang yang ingin menjal mobil bekas mereka, waktu yang dibutuhkan untuk meneliti harga yang kompetitif dapat bervariasi, mulai dari  beberapa jam hinggaa beberapa hari. ini melibatkan pengecakan daftar harga online, kunjungan ke dealer, dan pertimbangan faktor-faktor seperti merk, model, tahun, jarak, tempuh, kondisi, dan permintaan.[^1] [^2] [^3]

2. **mengidentifikasi faktor-faktor apa saja yang mempengaruhi harga mobil pada saat tertentu memerlukan banyak waktu jika dilakukan secara manual.** seperti halnya menentukan harga, mencari tahu faktor-faktor apa saja yang dapat mempengaruhi harga mobil sangatlah memakan waktu. dan tidak semua seller mempunya waktu ataupun kemauan untuk me-researchnya

### Goals
1. **Accurately Predict the price of used cars** : membuat model dapat membantu penjualan mobil bekas dengan menetapkan range harga yang kompetitif dan akurat / 20-25% MAPE
2. **Understanding Key Factors** : mengidentifikasi faktor-faktor penting yang dapat mempengaruhi harga mobil bekas dijual

### Business Questions
1. **berapa prediksi harga dari sebuah mobil bekas dengan feature tertentu?** 
2. **feature mana yang berpengaruh dalam menetapkan harga mobil bekas?**
3. **bagaimana price distribution dari mobil bekas?**
4. **apa saja popular brand dari setiap region categorynya?**
5. **apa saja type mobil yang populer dari setiap region categorynya?**
6. **gear and fuel type apa saja yang populer?**
7. **bagaimana relasi antara mileage dan age dari mobil bekas dengan popularitynya?**

### Evaluation Metric
- MedAE : Median Absolute Error adalah metric yang robust kepada outlier, loss dihitung dengan mengambil median dari semua absoulute differences antara target dan prediksi. MedAE dipilih karena data yang akan diprediksi mempunyai outlier yang cukup signifikan.    
- MAE : Mean Absolute Error adalah metric yang digunakan untuk memprediksi range harga. range harga yaitu harga prediksi machine learning ditambang dan dikurangi MAE.
- MAPE : Mean Absolute Percentage Error adalah bentuk persentase dari MAE, yang digunakan untuk merepresentasikan error rate pada prediksi machine learningnya

### Project Limitation
- dataset row yang memiliki value "True" pada column "negotiable" di drop karena memiliki nilai value 0 pada column "Price"
- dataset berisi mobil bekas dari tahun 2003 - 2021
- dataset berisi mobil bekas dengan ukuran engine size 1.0 - 8.0
- dataset berisi mobil bekas dengan mileage 100 - 432000
- dataset berisi mobil bekasi dengan harga 4000 - 1150000
- dataset tidak memiliki population size untuk column regionnya, jadi diharuskan untuk mencari populasi untuk setiap valuenya dari external source, untuk men-categorisasikannya. 
- Hardware Machine Learning

### Data Features and Description
nama dataset : Saudi Arabia Used Cars Dataset

data set berisi mobil bekas sebanyak 8035 records yang diambil dari syarah.com. setiap row merepresentasikan sebuah mobil bekas dengan informasi mengenai brand name, model, manufacturing year, origin, the color of the car, options, capacity of the engine, type of fuel, transmission type, the mileage that the car covered, region price, and negotiable

| Column | Data Type | Description |
| --- | --- | --- |
| Make | str - Nominal | nama brand dari mobil |
| Type | str - Nominal | jenis dari mobil |
| Year | int - Interval| tahun produksi mobil |
| Origin | str - Nominal| asal mobil |
| Color | str - Nominal| warna dari mobil |
| Options | str - Ordinal| kelengkapan opsi yang ada pada mobil |
| Engine_Size | float - Ratio| ukuran mesin yang digunakan oleh mobil |
| Fuel_Type | str - Nominal | type bahan bakar yang digunakan oleh mobil |
| Mileage | int - Ratio | jarak tempuh mobil |
| Region | str - Nominal| wilayah tempat mobil dijual |
| Price | int - Ratio| harga mobil |
| Negotiable | bool - Bool| negosiasi harga mobil |



[^1]:https://www.kenresearch.com/industry-reports/singapore-use-car-market  
[^2]:https://lotlinx.com/used-car-inventory-trends/   
[^3]:https://markwideresearch.com/singapore-used-car-market/   
