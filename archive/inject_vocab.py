import random
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIR / "dataset" / "dataset_pmi_master.csv"
ANCHOR_COUNT_PER_CLASS = 50
RANDOM_SEED = 2026


POSITIVE_ANCHORS = [
    "Pelayanan bagus dan petugas ramah saat membantu pendonor.",
    "Proses donor mantap, cepat, dan sangat memuaskan.",
    "Fasilitas keren dan suasana PMI terasa nyaman.",
    "Kinerja petugas luar biasa, profesional, dan sangat baik.",
    "Pelayanannya juara, top, dan recommended untuk donor rutin.",
    "Saya puas karena prosesnya cepat dan petugasnya ramah.",
    "Pengalaman donor sangat baik, lancar, dan memuaskan.",
    "Pelayanan bagus sekali, petugas profesional dan komunikatif.",
    "Tempatnya keren, bersih, dan membuat pendonor nyaman.",
    "PMI memberikan layanan mantap dengan proses yang cepat.",
    "Petugas ramah dan hasil pelayanan sangat memuaskan.",
    "Pelayanan luar biasa, tertib, cepat, dan profesional.",
    "Saya merekomendasikan PMI karena pelayanannya top.",
    "Proses pendaftaran cepat dan petugas bekerja sangat baik.",
    "Fasilitas bagus dan pelayanan donor terasa juara.",
    "Petugasnya ramah, sopan, dan sangat membantu.",
    "Pelayanan profesional dengan hasil yang memuaskan.",
    "Pengalaman hari ini mantap, cepat, dan tidak mengecewakan.",
    "Tempat donor keren dengan layanan yang sangat baik.",
    "Saya puas, pelayanan cepat, ramah, dan recommended.",
    "Kualitas pelayanan bagus dan petugas sangat profesional.",
    "Layanan PMI luar biasa, prosesnya cepat dan tertata.",
    "Fasilitas top, petugas ramah, dan pelayanan memuaskan.",
    "Donor darah di sini terasa nyaman karena layanan sangat baik.",
    "Petugas cepat tanggap dan memberikan bantuan dengan ramah.",
    "Pelayanan mantap sejak pendaftaran sampai selesai donor.",
    "Saya sangat merekomendasikan tempat ini karena profesional.",
    "Suasana nyaman, fasilitas keren, dan pelayanan bagus.",
    "Prosesnya cepat sekali dan petugasnya sangat ramah.",
    "Pelayanan juara dengan tenaga medis yang profesional.",
    "Hasil pelayanan sangat memuaskan dan alurnya jelas.",
    "PMI memberikan pengalaman donor yang luar biasa baik.",
    "Petugas sopan, komunikatif, dan bekerja dengan sangat baik.",
    "Layanan top, ruang bersih, dan proses donor cepat.",
    "Saya puas dengan pelayanan yang mantap dan teratur.",
    "Tempat ini keren dan recommended untuk pendonor baru.",
    "Petugas ramah ketika menjelaskan prosedur donor darah.",
    "Pelayanan cepat, fasilitas bagus, dan pengalaman memuaskan.",
    "Kinerja tim PMI sangat profesional dan luar biasa.",
    "Proses donor lancar, nyaman, dan terasa sangat baik.",
    "Pelayanan bagus tanpa kendala dari awal hingga akhir.",
    "Saya mendapat bantuan cepat dari petugas yang ramah.",
    "Fasilitas top dan kualitas pelayanan sangat memuaskan.",
    "Pengalaman positif, petugas profesional, dan alur cepat.",
    "PMI Cirebon juara dalam memberikan pelayanan donor.",
    "Pelayanan sangat baik, tertib, bersih, dan nyaman.",
    "Saya puas dan akan kembali karena layanannya recommended.",
    "Petugas mantap, ramah, serta sigap membantu pendonor.",
    "Proses cepat dengan fasilitas keren dan layanan profesional.",
    "Pelayanan luar biasa bagus dan sangat memuaskan hari ini.",
]

NEGATIVE_ANCHORS = [
    "Pelayanan jelek dan petugas lambat menangani antrean.",
    "Kondisi ruang tunggu kotor dan sangat mengecewakan.",
    "Proses administrasi buruk, parah, dan tidak tertib.",
    "Antrean panjang membuat pelayanan menjadi sangat lambat.",
    "Petugas kasar dan tidak profesional saat menjawab pertanyaan.",
    "Saya kecewa karena sistem pendaftaran lelet dan membingungkan.",
    "Ruang tunggu bau, kotor, dan tidak nyaman bagi pendonor.",
    "Pelayanan buruk sekali, saya menyesal datang ke sini.",
    "Petugas sombong dan tidak ramah ketika melayani masyarakat.",
    "Kinerja loket parah karena antrean panjang tidak terkendali.",
    "Proses donor lambat dan respons petugas sangat mengecewakan.",
    "Fasilitas kotor, bau, dan kualitas pelayanan jelek.",
    "Saya kecewa dengan pelayanan yang tidak profesional.",
    "Sistem antrean lelet sehingga waktu tunggu menjadi panjang.",
    "Petugas kasar saat menerima keluhan dan tidak memberi solusi.",
    "Pelayanan parah, buruk, dan jauh dari harapan pendonor.",
    "Kondisi toilet kotor serta menimbulkan bau tidak sedap.",
    "Antrean panjang dibiarkan tanpa arahan dari petugas.",
    "Pengalaman donor sangat mengecewakan dan membuat saya menyesal.",
    "Petugas sombong, lambat, dan tidak profesional.",
    "Fasilitas jelek dan ruang pelayanan terasa sangat kotor.",
    "Proses pendaftaran buruk karena komputer lelet dan sering gagal.",
    "Saya kecewa menunggu lama untuk pelayanan yang sangat lambat.",
    "Suasana ruangan bau dan petugas melayani dengan kasar.",
    "Pelayanan tidak profesional, antrean panjang, dan informasi tidak jelas.",
    "Hari ini pelayanan parah, jelek, dan tidak memuaskan.",
    "Petugas tidak ramah, bahkan terkesan sombong kepada pendonor.",
    "Ruang observasi kotor dan penanganan setelah donor sangat lambat.",
    "Saya menyesal datang karena pelayanan buruk dan tidak tertib.",
    "Antrean panjang, sistem lelet, dan petugas tidak sigap.",
    "Kualitas kebersihan jelek karena banyak area kotor dan bau.",
    "Proses administrasi sangat mengecewakan dan tidak profesional.",
    "Pelayanan lambat padahal antrean sudah menunggu sejak pagi.",
    "Petugas kasar ketika menjelaskan aturan kepada pendonor.",
    "Fasilitas buruk dan kondisi ruangan parah sekali.",
    "Saya kecewa karena keluhan tidak ditanggapi dengan baik.",
    "Layanan PMI hari ini jelek, lelet, dan membuang waktu.",
    "Petugas sombong dan membuat pendonor merasa tidak dihargai.",
    "Antrean panjang tanpa sistem yang jelas sangat mengecewakan.",
    "Toilet bau dan kotor, pelayanan juga tidak profesional.",
    "Pengalaman donor buruk, lambat, dan membuat saya menyesal.",
    "Pelayanan parah karena petugas kurang ramah dan tidak sigap.",
    "Sistem komputer lelet sehingga pendaftaran menjadi sangat lambat.",
    "Kondisi ruang tunggu jelek, kotor, dan tidak nyaman.",
    "Saya kecewa dengan sikap kasar petugas di loket.",
    "Pelayanan tidak profesional dan antrean panjang dibiarkan.",
    "Pengalaman hari ini sangat buruk serta mengecewakan.",
    "Fasilitas bau, kotor, dan proses pelayanan terlalu lambat.",
    "Saya menyesal memilih tempat ini karena layanannya parah.",
    "Petugas tidak ramah, proses lelet, dan kualitasnya jelek.",
]

NEUTRAL_ANCHORS = [
    "Pelayanan standar dan proses berjalan seperti biasa.",
    "Pengalaman donor biasa saja, tidak ada yang spesial.",
    "Proses berjalan normal sesuai prosedur yang berlaku.",
    "Fasilitas cukup dan pelayanan terasa lumayan.",
    "Kunjungan hari ini wajar seperti kunjungan sebelumnya.",
    "Pelayanan cukup untuk kebutuhan donor rutin.",
    "Semuanya normal, tidak ada masalah berarti hari ini.",
    "Pengalaman biasa saja dengan fasilitas yang cukup.",
    "Prosedur standar dan berjalan sesuai antrean.",
    "Pelayanan lumayan, tidak terlalu cepat maupun lambat.",
    "Tidak ada yang spesial dari kunjungan donor kali ini.",
    "Kondisi ruangan wajar dan proses berjalan normal.",
    "Pelayanan cukup memadai untuk donor darah rutin.",
    "Semua berjalan seperti biasa tanpa perubahan berarti.",
    "Pengalaman standar, fasilitas cukup, dan proses normal.",
    "Pelayanan lumayan sesuai dengan harapan biasa.",
    "Kunjungan berjalan wajar tanpa kejadian khusus.",
    "Tidak ada yang spesial, semuanya sesuai prosedur.",
    "Proses donor normal dan fasilitasnya cukup.",
    "Pelayanan biasa saja seperti pengalaman sebelumnya.",
    "Kualitas layanan standar dan alur pendaftaran normal.",
    "Saya merasa cukup dengan pelayanan hari ini.",
    "Suasana wajar dan proses donor berjalan seperti biasa.",
    "Pelayanan lumayan, tidak ada keluhan penting.",
    "Kunjungan rutin berjalan normal dan sesuai aturan.",
    "Fasilitas standar dengan pelayanan yang cukup.",
    "Pengalaman donor tidak spesial, tetapi berjalan normal.",
    "Semua cukup tertib dan prosesnya wajar.",
    "Layanan hari ini biasa saja seperti biasanya.",
    "Pelayanan normal dan fasilitas cukup untuk pendonor.",
    "Tidak ada perubahan, proses berjalan standar.",
    "Kondisi PMI wajar dan pelayanannya lumayan.",
    "Saya datang rutin, pelayanan masih seperti biasa.",
    "Prosedur cukup jelas dan berlangsung normal.",
    "Pengalaman standar tanpa hal yang spesial.",
    "Pelayanan cukup baik dalam batas yang wajar.",
    "Kunjungan biasa dengan proses donor yang normal.",
    "Fasilitas lumayan dan alurnya sesuai prosedur.",
    "Semua berjalan standar, tidak ada masalah besar.",
    "Pelayanan wajar dan waktu tunggunya biasa saja.",
    "Tidak ada yang istimewa, proses tetap normal.",
    "Fasilitas cukup untuk mendukung donor rutin.",
    "Kunjungan berlangsung seperti biasa dan cukup lancar.",
    "Layanan standar dengan kondisi ruangan yang wajar.",
    "Saya menilai pelayanan hari ini lumayan saja.",
    "Proses normal, tidak terlalu berbeda dari sebelumnya.",
    "Pelayanan cukup dan tidak ada kejadian khusus.",
    "Pengalaman donor biasa saja tetapi tetap berjalan.",
    "Semuanya wajar sesuai standar pelayanan.",
    "Kunjungan ini cukup biasa dengan pelayanan yang normal.",
]


def inject_anchors(dataframe: pd.DataFrame, label: str, reviews: list[str], rng: random.Random) -> None:
    indexes = dataframe.index[dataframe["sentimen"].eq(label)].tolist()
    if len(indexes) < ANCHOR_COUNT_PER_CLASS:
        raise ValueError(f"Data berlabel {label} kurang dari {ANCHOR_COUNT_PER_CLASS} baris.")
    if len(reviews) < ANCHOR_COUNT_PER_CLASS:
        raise ValueError(f"Anchor untuk label {label} belum mencapai 50 ulasan.")

    selected_indexes = rng.sample(indexes, ANCHOR_COUNT_PER_CLASS)
    selected_reviews = rng.sample(reviews, ANCHOR_COUNT_PER_CLASS)
    for index, review in zip(selected_indexes, selected_reviews):
        dataframe.at[index, "ulasan"] = review


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_FILE}")

    dataframe = pd.read_csv(DATA_FILE)
    required_columns = {"tanggal", "ulasan", "sentimen"}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Kolom wajib tidak ditemukan: {missing}")

    rng = random.Random(RANDOM_SEED)
    inject_anchors(dataframe, "Positif", POSITIVE_ANCHORS, rng)
    inject_anchors(dataframe, "Negatif", NEGATIVE_ANCHORS, rng)
    inject_anchors(dataframe, "Netral", NEUTRAL_ANCHORS, rng)
    dataframe.to_csv(DATA_FILE, index=False)

    print("Berhasil menginjeksi 150 anchor vocabulary:")
    print("- Positif: 50")
    print("- Negatif: 50")
    print("- Netral: 50")
    print(f"Total baris tetap: {len(dataframe)}")
    print(f"Dataset tersimpan: {DATA_FILE}")


if __name__ == "__main__":
    main()
