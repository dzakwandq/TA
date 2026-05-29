import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_gauges/gauges.dart';
import 'package:file_picker/file_picker.dart';

void main() {
  runApp(const HemoSnapApp());
}

class HemoSnapApp extends StatelessWidget {
  const HemoSnapApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'HemoSnap',
      theme: ThemeData(
        scaffoldBackgroundColor: Colors.white,
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFD32F2F)),
        useMaterial3: true,
        fontFamily: 'Roboto', // Default font mirip sans-serif bersih
      ),
      home: const BerandaScreen(),
    );
  }
}

class BerandaScreen extends StatefulWidget {
  const BerandaScreen({super.key});

  @override
  State<BerandaScreen> createState() => _BerandaScreenState();
}

class _BerandaScreenState extends State<BerandaScreen> {
  int _selectedIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Row(
          children: [
            const Icon(Icons.water_drop, color: Color(0xFFD32F2F)),
            const SizedBox(width: 8),
            const Text(
              'HemoSnap',
              style: TextStyle(
                color: Colors.black87,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.notifications_none, color: Colors.black87),
            onPressed: () {},
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Sapaan
            const Text(
              'Halo, Dzakwan! 👋',
              style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text(
              'Pantau kadar hemoglobinmu\ndengan mudah dan cepat.',
              style: TextStyle(color: Colors.grey, fontSize: 14),
            ),
            const SizedBox(height: 40),

            // Tombol Utama (Lingkaran Merah)
            Center(
              child: GestureDetector(
                onTap: () {
                  // Nanti di sini kita arahkan ke Halaman Panduan Unggah (Screen 2)
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                        builder: (context) => const PanduanUnggahScreen()
                    ),
                  );
                },
                child: Container(
                  width: 220,
                  height: 220,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: const Color(0xFFD32F2F).withOpacity(0.1),
                  ),
                  child: Center(
                    child: Container(
                      width: 170,
                      height: 170,
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: const Color(0xFFD32F2F),
                        boxShadow: [
                          BoxShadow(
                            color: const Color(0xFFD32F2F).withOpacity(0.4),
                            blurRadius: 20,
                            spreadRadius: 5,
                          ),
                        ],
                      ),
                      child: const Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(Icons.camera_alt, color: Colors.white, size: 40),
                          SizedBox(height: 10),
                          Text(
                            'Mulai Skrining Baru\n(RAW/DNG)',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: Colors.white,
                              fontWeight: FontWeight.bold,
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 40),

            // Riwayat Terakhir
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                const Text(
                  'Riwayat Terakhir',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
                ),
                TextButton(
                  onPressed: () {},
                  child: const Text(
                    'Lihat Semua',
                    style: TextStyle(color: Color(0xFFD32F2F)),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 10),

            // Kartu Riwayat
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.grey.shade200),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.05),
                    blurRadius: 10,
                    spreadRadius: 2,
                  ),
                ],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(Icons.calendar_today, size: 14, color: Colors.grey.shade500),
                          const SizedBox(width: 5),
                          Text(
                            '12 Mei 2026 • 09:32',
                            style: TextStyle(color: Colors.grey.shade600, fontSize: 12),
                          ),
                        ],
                      ),
                      const SizedBox(height: 15),
                      const Text(
                        'Normal',
                        style: TextStyle(
                          color: Colors.green,
                          fontWeight: FontWeight.bold,
                          fontSize: 18,
                        ),
                      ),
                      const Text(
                        '14.5 g/dL',
                        style: TextStyle(
                          color: Colors.black87,
                          fontWeight: FontWeight.bold,
                          fontSize: 16,
                        ),
                      ),
                    ],
                  ),

                  // Mini Gauge Chart dari library Syncfusion
                  SizedBox(
                    width: 80,
                    height: 80,
                    child: SfRadialGauge(
                      axes: <RadialAxis>[
                        RadialAxis(
                          minimum: 0,
                          maximum: 100,
                          showLabels: false,
                          showTicks: false,
                          axisLineStyle: const AxisLineStyle(
                            thickness: 0.2,
                            cornerStyle: CornerStyle.bothCurve,
                            color: Color.fromARGB(255, 224, 224, 224),
                            thicknessUnit: GaugeSizeUnit.factor,
                          ),
                          pointers: const <GaugePointer>[
                            RangePointer(
                              value: 70, // Contoh nilai
                              cornerStyle: CornerStyle.bothCurve,
                              width: 0.2,
                              sizeUnit: GaugeSizeUnit.factor,
                              color: Colors.green,
                            )
                          ],
                        )
                      ],
                    ),
                  ),
                  const Icon(Icons.chevron_right, color: Colors.grey),
                ],
              ),
            ),
          ],
        ),
      ),

      // Bottom Navigation Bar
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _selectedIndex,
        selectedItemColor: const Color(0xFFD32F2F),
        unselectedItemColor: Colors.grey,
        onTap: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Beranda'),
          BottomNavigationBarItem(icon: Icon(Icons.receipt_long), label: 'Riwayat'),
          BottomNavigationBarItem(icon: Icon(Icons.person), label: 'Profil'),
        ],
      ),
    );
  }
}

// ==========================================
// SCREEN 2: PANDUAN UNGGAH
// ==========================================
class PanduanUnggahScreen extends StatelessWidget {
  const PanduanUnggahScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black87, size: 20),
          onPressed: () {
            // Logika untuk kembali ke halaman sebelumnya
            Navigator.pop(context);
          },
        ),
        title: const Text(
          'Panduan Unggah',
          style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 18),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.info_outline, color: Colors.black87),
            onPressed: () {},
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Placeholder Gambar Ilustrasi
            Container(
              height: 180,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.grey.shade300, style: BorderStyle.solid),
              ),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.image_outlined, size: 50, color: Colors.grey.shade400),
                  const SizedBox(height: 10),
                  Text(
                    '[Gambar Ilustrasi Mata & Grey Card]',
                    style: TextStyle(color: Colors.grey.shade500, fontSize: 12),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            const Text(
              'Pastikan foto memenuhi syarat berikut:',
              style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.black87),
            ),
            const SizedBox(height: 16),

            // Daftar Checklist Persyaratan
            _buildChecklistItem(
              icon: Icons.raw_on,
              title: 'Gunakan File RAW/DNG',
              subtitle: 'Format file harus RAW atau DNG.',
            ),
            _buildChecklistItem(
              icon: Icons.wb_sunny_outlined,
              title: 'Cahaya Alami, Tidak Terlalu Terang/Gelap',
              subtitle: 'Ambil foto di tempat terang dengan cahaya alami, hindari penggunaan flash.',
            ),
            _buildChecklistItem(
              icon: Icons.style_outlined, // Ikon kartu pengganti grey card
              title: 'Grey Card Terlihat Jelas',
              subtitle: 'Pastikan grey card berada di samping mata dan terlihat jelas.',
            ),
            _buildChecklistItem(
              icon: Icons.center_focus_strong_outlined,
              title: 'Fokus Camera pada Mata',
              subtitle: 'Pastikan konjungtiva (bagian dalam kelopak mata) fokus dan tidak buram.',
            ),
          ],
        ),
      ),
      // Area Tombol Bawah (Statis agar selalu terlihat)
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              SizedBox(
                width: double.infinity,
                height: 50,
                child: ElevatedButton.icon(
                  onPressed: () async{
                    // TODO: Nanti kita tambahkan logika buka galeri di sini
                    FilePickerResult? result = await FilePicker.platform.pickFiles(
                      type: FileType.custom,
                      allowedExtensions: ['dng', 'raw'],
                    );

                    //cek udah pilih file ato belum
                    if (result != null) {
                      //ambil data dari file yang dipilih
                      String fileName = result.files.first.name;
                      double fileSize = result.files.first.size / (1024 * 1024); //convert byte ke mb

                      //notif kecil udh pick file
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text('Memproses file: $fileName (${fileSize.toStringAsFixed(2)} MB)'),
                              backgroundColor: Colors.blue,
                              behavior: SnackBarBehavior.floating,
                              duration: const Duration(seconds: 1),
                            ),
                        );

                        //langsung pindah ke bagian hasil skrining
                        Navigator.push(
                            context,
                            MaterialPageRoute(
                                builder: (context) => HasilSkriningScreen()
                            ),
                        );

                        //TODO: code pindah ke halaman "Loading Analisis" di sini

                      }
                    } else {
                      //kalau user nutup jendela galeri dan ga milih foto
                      if (context.mounted) {
                        ScaffoldMessenger.of(context).showSnackBar(
                            const SnackBar(
                              content: Text('Pemilian File Dibatalkan'),
                              backgroundColor: Colors.grey,
                              behavior: SnackBarBehavior.floating,
                            ),
                        );
                      }
                    }
                  },
                  icon: const Icon(Icons.photo_library_outlined, color: Colors.white),
                  label: const Text(
                    'Pilih Foto DNG dari Galeri',
                    style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
                  ),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFD32F2F),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                ),
              ),
              const SizedBox(height: 12),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.lock_outline, size: 14, color: Colors.grey.shade500),
                  const SizedBox(width: 6),
                  Text(
                    'Foto nanti diproses di server.',
                    style: TextStyle(fontSize: 11, color: Colors.grey.shade500),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }

  // Widget kustom pembantu agar kode rapi
  Widget _buildChecklistItem({required IconData icon, required String title, required String subtitle}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.grey.shade200),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0xFFD32F2F).withOpacity(0.08),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: const Color(0xFFD32F2F)),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14, color: Colors.black87),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  style: TextStyle(fontSize: 12, color: Colors.grey.shade600),
                ),
              ],
            ),
          ),
          const SizedBox(width: 10),
          const Icon(Icons.check_circle, color: Colors.green),
        ],
      ),
    );
  }
}

// ==========================================
// SCREEN 3: HASIL SKRINING
// ==========================================
class HasilSkriningScreen extends StatelessWidget {
  const HasilSkriningScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.grey.shade50, // Latar belakang sedikit abu-abu agar kartu putih menonjol
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios, color: Colors.black87, size: 20),
          onPressed: () => Navigator.pop(context),
        ),
        title: const Text(
          'Hasil Skrining',
          style: TextStyle(color: Colors.black87, fontWeight: FontWeight.bold, fontSize: 18),
        ),
        centerTitle: true,
        actions: [
          IconButton(
            icon: const Icon(Icons.more_horiz, color: Colors.black87),
            onPressed: () {},
          ),
        ],
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // --- BAGIAN A: Visualisasi ---
            const Text(
              'Visualisasi Hasil Segmentasi',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.black87),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.grey.shade200),
              ),
              child: Row(
                children: [
                  Expanded(
                    child: _buildImagePlaceholder('Foto Konjungtiva\n(Corrected)', Icons.remove_red_eye),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: _buildImagePlaceholder('Hasil Masking\n(UNBCSM)', Icons.contrast, isDark: true),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 24),

            // --- BAGIAN B: Status & Analisis ---
            const Text(
              'Status & Analisis',
              style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.black87),
            ),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.red.shade100, width: 2), // Border merah muda
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Badge Peringatan Merah
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    decoration: BoxDecoration(
                      color: const Color(0xFFD32F2F),
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.warning_amber_rounded, color: Colors.white, size: 16),
                        SizedBox(width: 6),
                        Text(
                          'INDIKASI ANEMIA',
                          style: TextStyle(color: Colors.white, fontWeight: FontWeight.bold, fontSize: 12),
                        ),
                      ],
                    ),
                  ),
                  const SizedBox(height: 16),

                  // Nilai dan Grafik Gauge
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    crossAxisAlignment: CrossAxisAlignment.end,
                    children: [
                      Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            'Estimasi Kadar Hemoglobin',
                            style: TextStyle(fontSize: 12, color: Colors.black87),
                          ),
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.baseline,
                            textBaseline: TextBaseline.alphabetic,
                            children: [
                              const Text(
                                '8.2',
                                style: TextStyle(fontSize: 48, fontWeight: FontWeight.bold, color: Color(0xFFD32F2F)),
                              ),
                              const SizedBox(width: 4),
                              Text(
                                'g/dL',
                                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.grey.shade700),
                              ),
                            ],
                          ),
                          const Text(
                            'Rendah (< 12.0 g/dL)',
                            style: TextStyle(fontSize: 12, color: Colors.grey),
                          ),
                        ],
                      ),

                      // Gauge Setengah Lingkaran
                      SizedBox(
                        width: 110,
                        height: 70,
                        child: SfRadialGauge(
                          axes: <RadialAxis>[
                            RadialAxis(
                              minimum: 0,
                              maximum: 20,
                              startAngle: 180,
                              endAngle: 0,
                              showLabels: true,
                              showTicks: false,
                              labelOffset: 10,
                              axisLabelStyle: const GaugeTextStyle(fontSize: 10, fontWeight: FontWeight.bold),
                              interval: 10,
                              axisLineStyle: const AxisLineStyle(
                                thickness: 0.2,
                                cornerStyle: CornerStyle.bothCurve,
                                color: Color(0xFFE0E0E0),
                                thicknessUnit: GaugeSizeUnit.factor,
                              ),
                              ranges: <GaugeRange>[
                                GaugeRange(
                                  startValue: 0,
                                  endValue: 12,
                                  color: const Color(0xFFD32F2F),
                                  startWidth: 0.2,
                                  endWidth: 0.2,
                                  sizeUnit: GaugeSizeUnit.factor,
                                ),
                                GaugeRange(
                                  startValue: 12,
                                  endValue: 20,
                                  color: Colors.green,
                                  startWidth: 0.2,
                                  endWidth: 0.2,
                                  sizeUnit: GaugeSizeUnit.factor,
                                ),
                              ],
                              pointers: const <GaugePointer>[
                                NeedlePointer(
                                  value: 8.2, // Nilai dummy hasil prediksi modelmu nanti
                                  needleLength: 0.6,
                                  needleStartWidth: 1,
                                  needleEndWidth: 4,
                                  knobStyle: KnobStyle(knobRadius: 0.08, sizeUnit: GaugeSizeUnit.factor, color: Colors.black),
                                )
                              ],
                            )
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 20),

                  // Kotak Rekomendasi Dokter
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.grey.shade50,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: Colors.grey.shade300),
                    ),
                    child: Row(
                      children: [
                        const CircleAvatar(
                          backgroundColor: Color(0xFFE3F2FD),
                          child: Icon(Icons.medical_services, color: Color(0xFF1565C0)),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: RichText(
                            text: const TextSpan(
                              style: TextStyle(fontSize: 12, color: Colors.black87, height: 1.4),
                              children: [
                                TextSpan(text: 'Kadar hemoglobin Anda berada di bawah normal. '),
                                TextSpan(text: 'Konsultasikan dengan Dokter', style: TextStyle(fontWeight: FontWeight.bold, color: Color(0xFFD32F2F))),
                                TextSpan(text: ' untuk evaluasi lebih lanjut.'),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
      // Tombol Simpan & Kembali
      bottomNavigationBar: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(20.0),
          child: SizedBox(
            width: double.infinity,
            height: 50,
            child: ElevatedButton.icon(
              onPressed: () {
                // Perintah popUntil ini akan menutup semua halaman dan kembali ke BerandaScreen (Halaman Pertama)
                Navigator.popUntil(context, (route) => route.isFirst);
              },
              icon: const Icon(Icons.save_alt, color: Colors.white),
              label: const Text(
                'Simpan & Kembali ke Beranda',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Colors.white),
              ),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFD32F2F),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
              ),
            ),
          ),
        ),
      ),
    );
  }

  // Widget pembantu untuk kotak placeholder gambar (kiri mata, kanan mask hitam)
  Widget _buildImagePlaceholder(String label, IconData icon, {bool isDark = false}) {
    return Column(
      children: [
        Text(label, textAlign: TextAlign.center, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600)),
        const SizedBox(height: 8),
        Container(
          height: 120,
          width: double.infinity,
          decoration: BoxDecoration(
            color: isDark ? Colors.black87 : Colors.grey.shade200, // Hitam untuk mask UNBCSM
            borderRadius: BorderRadius.circular(12),
          ),
          child: Icon(icon, size: 40, color: isDark ? Colors.white54 : Colors.grey.shade400),
        ),
      ],
    );
  }
}