import cv2
import os
import time
import json
import numpy as np
from collections import deque
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RTSP_URL = os.environ.get('OPET_RTSP_URL', '')
MODEL_DOSYASI = os.environ.get('OPET_MODEL_PATH', os.path.join(BASE_DIR, 'models', 'pose-model.pt'))
AYARLAR_DOSYASI = os.environ.get('OPET_CONFIG_PATH', os.path.join(BASE_DIR, 'config', 'settings.json'))
OUTPUT_DIR = os.environ.get('OPET_OUTPUT_DIR', os.path.join(BASE_DIR, 'data'))
LOG_KLASORU = os.path.join(OUTPUT_DIR, 'logs')
KAYIT_KLASORU = os.path.join(OUTPUT_DIR, 'recordings')
SATIS_KLASORU = os.path.join(OUTPUT_DIR, 'sales')
SCREENSHOT_KLASORU = os.path.join(OUTPUT_DIR, 'screenshots')
FPS_SINIRI = int(os.environ.get('OPET_FPS_LIMIT', '5'))
BOSLUK_SURESI = 10.0
ON_BUFFER_SN = 20
DEVICE = os.environ.get('OPET_DEVICE', 'cuda')

if not RTSP_URL:
    raise RuntimeError('OPET_RTSP_URL must be set to a private, local camera stream.')
if not os.path.exists(MODEL_DOSYASI):
    raise FileNotFoundError(f'Model file not found: {MODEL_DOSYASI}')
os.makedirs(LOG_KLASORU, exist_ok=True)
os.makedirs(KAYIT_KLASORU, exist_ok=True)
os.makedirs(SATIS_KLASORU, exist_ok=True)
os.makedirs(SCREENSHOT_KLASORU, exist_ok=True)
global_alisveris = {
    'best_prio': 99, 
    'best_frame': None, 
    'event_name': '', 
    'last_event_time': 0,
    'start_time': 0,
    'ss_guncelle': False
}

LOG_DOSYASI = os.path.join(LOG_KLASORU, 'kahve_' + time.strftime('%Y-%m-%d') + '.txt')
SATIS_DOSYASI = os.path.join(SATIS_KLASORU, 'satis_' + time.strftime('%Y-%m-%d') + '.txt')

def log(kategori, mesaj):
    satir = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] [{kategori}] {mesaj}"
    print(satir)
    with open(LOG_DOSYASI, 'a', encoding='utf-8') as f:
        f.write(satir + '\n')

def satis_kaydet(islem_no):
    satir = f"{islem_no}. Kahve -> Saat: {time.strftime('%H:%M:%S')}"
    with open(SATIS_DOSYASI, 'a', encoding='utf-8') as f:
        f.write(satir + '\n')

log('SISTEM', '=' * 60)
log('SISTEM', 'SADECE BARDAK TAKIBI - Basladi (YOLO26)')
log('SISTEM', '=' * 60)

print('YOLO26 Yapay Zekasi yukleniyor, lutfen bekleyin...')
model = YOLO(MODEL_DOSYASI)
model.to(DEVICE)
gstreamer_pipeline = (
    f'rtspsrc location="{RTSP_URL}" protocols=tcp latency=0 '
    '! rtph265depay ! h265parse ! avdec_h265 '
    '! videoconvert ! video/x-raw,format=BGR ! appsink drop=true sync=false'
)
cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    log('HATA', 'GStreamer ile kameraya baglanilmadi!')
    exit()
current_frame_jpeg = b''

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global current_frame_jpeg
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    if current_frame_jpeg:
                        self.wfile.write(b'--frame\r\n')
                        self.send_header('Content-type', 'image/jpeg')
                        self.send_header('Content-length', len(current_frame_jpeg))
                        self.end_headers()
                        self.wfile.write(current_frame_jpeg)
                        self.wfile.write(b'\r\n')
                    time.sleep(0.04)
            except Exception:
                pass

def start_server():
    server = HTTPServer(('0.0.0.0', 5000), StreamHandler)
    server.serve_forever()

server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()
log('SISTEM', 'Canli Yayin Sunucusu Basladi! Tarayicinizdan http://localhost:5000 adresine gidebilirsiniz.')
if os.path.exists(AYARLAR_DOSYASI):
    with open(AYARLAR_DOSYASI, 'r') as f:
        d = json.load(f)
    FINCAN_X1, FINCAN_Y1, FINCAN_X2, FINCAN_Y2 = d['FINCAN_X1'], d['FINCAN_Y1'], d['FINCAN_X2'], d['FINCAN_Y2']
    try:
        FIS_X1, FIS_Y1, FIS_X2, FIS_Y2 = d['FIS_X1'], d['FIS_Y1'], d['FIS_X2'], d['FIS_Y2']
    except:
        FIS_X1, FIS_Y1, FIS_X2, FIS_Y2 = 0, 0, 10, 10
    try:
        KISI_X1, KISI_Y1, KISI_X2, KISI_Y2 = d['KISI_X1'], d['KISI_Y1'], d['KISI_X2'], d['KISI_Y2']
    except:
        KISI_X1, KISI_Y1, KISI_X2, KISI_Y2 = 0, 0, 1920, 1080
    try:
        EL_X1, EL_Y1, EL_X2, EL_Y2 = d['EL_X1'], d['EL_Y1'], d['EL_X2'], d['EL_Y2']
    except:
        EL_X1, EL_Y1, EL_X2, EL_Y2 = 0, 0, 10, 10
else:
    print('HATA: ayarlar.json bulunamadi. Lutfen once alan_sec.py calistirin.')
    exit()
kahve_sayisi = 0
kisi_sayisi = 0
fis_sayisi = 0
bilek_sayisi = 0
islem_aktif = False
fis_aktif = False
bilek_aktif = False
kayit_aktif = False
ilk_kare_okundu = False
son_olay_zaman = 0
son_kahve_zaman = 0
son_fis_zaman = 0
son_bilek_zaman = 0

kisiler_takip = {}

fps_aralik = 1.0 / FPS_SINIRI
son_isleme_zaman = 0

on_buffer = deque(maxlen=ON_BUFFER_SN * FPS_SINIRI)
video_writer = None

try:
    cv2.namedWindow('Bardak Takibi', cv2.WINDOW_NORMAL)
except Exception:
    pass
while True:
    ret, frame = cap.read()
    if not ret:
        log('UYARI', 'Kamera baglantisi kesildi, GStreamer yeniden baglaniliyor...')
        time.sleep(1)
        cap.release()
        cap = cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
        continue

    if np.var(frame) < 50:
        continue
    current_time = time.time()
    if current_time - son_isleme_zaman < fps_aralik:
        continue
    son_isleme_zaman = current_time

    H, W = frame.shape[:2]
    fis_roi = frame[FIS_Y1:FIS_Y2, FIS_X1:FIS_X2]
    hsv_fis = cv2.cvtColor(fis_roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 50, 255])
    mask_fis = cv2.inRange(hsv_fis, lower_white, upper_white)
    beyaz_piksel_sayisi = cv2.countNonZero(mask_fis)
    toplam_alan = (FIS_X2 - FIS_X1) * (FIS_Y2 - FIS_Y1)
    min_giris_esigi = toplam_alan // 7
    roi_frame = frame[KISI_Y1:KISI_Y2, KISI_X1:KISI_X2]
    results = model.track(roi_frame, persist=True, verbose=False, conf=0.50, device=DEVICE, tracker="bytetrack.yaml")

    nesne_bolgede = False
    nesne_bolgede_musteri = False
    el_fiste = False
    el_fis_genis_alanda = False
    bilek_bolgede = False
    
    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        boxes[:, 0] += KISI_X1
        boxes[:, 2] += KISI_X1
        boxes[:, 1] += KISI_Y1
        boxes[:, 3] += KISI_Y1
        
        ids = results[0].boxes.id.cpu().numpy()
        
        keypoints_data = []
        if results[0].keypoints is not None and results[0].keypoints.xy is not None:
            keypoints_data = results[0].keypoints.xy.cpu().numpy()
            if len(keypoints_data) > 0:
                keypoints_data[:, :, 0] += KISI_X1
                keypoints_data[:, :, 1] += KISI_Y1
                
        confs = results[0].keypoints.conf.cpu().numpy() if results[0].keypoints is not None and results[0].keypoints.conf is not None else None

        for i, (box, person_id) in enumerate(zip(boxes, ids)):
            pid = int(person_id)
            
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            if pid not in kisiler_takip:
                for old_pid, data in kisiler_takip.items():
                    if old_pid in ids:
                        continue
                        
                    if 'son_merkez' in data:
                        gecen_sure_kayip = current_time - data.get('son_gorulme', current_time)
                        if 0 < gecen_sure_kayip < 2.0:
                            old_cx, old_cy = data['son_merkez']
                            mesafe = ((cx - old_cx)**2 + (cy - old_cy)**2) ** 0.5
                            if mesafe < 60:
                                pid = old_pid
                                break
            if pid not in kisiler_takip:
                kisiler_takip[pid] = {
                    'ilk_gorulme': current_time, 
                    'aktif_sure': 0.0,
                    'sayildi': False, 
                    'personel_mi': False,
                    'personel_skoru': 0,
                    'son_gorulme': current_time,
                    'son_merkez': (cx, cy)
                }
            else:
                kisiler_takip[pid]['son_gorulme'] = current_time
                kisiler_takip[pid]['son_merkez'] = (cx, cy)
                
            def islem_kaydet(prio, e_name):
                gecen_bosluk = current_time - global_alisveris['last_event_time']
                gecen_toplam = current_time - global_alisveris['start_time']
                if gecen_bosluk > 10.0 or gecen_toplam > 35.0:
                    if global_alisveris['best_frame'] is not None:
                        olay_sahibi_pid = global_alisveris.get('event_pid', -1)
                        if olay_sahibi_pid != -1 and olay_sahibi_pid in kisiler_takip and kisiler_takip[olay_sahibi_pid]['personel_mi']:
                            log('SCREENSHOT', f'Alisverisi baslatan kisi ({olay_sahibi_pid}) sonradan personel ciktigi icin SS IPTAL EDILDI.')
                        else:
                            ss_isim = f"Screenshot_Prio{global_alisveris['best_prio']}_{global_alisveris['event_name']}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            ss_path = os.path.join(SCREENSHOT_KLASORU, ss_isim)
                            cv2.imwrite(ss_path, global_alisveris['best_frame'])
                            sebep = "10s bosluk" if gecen_bosluk > 10.0 else "35s siniri"
                            log('SCREENSHOT', f'Alisveris tamamlandi ({sebep}). SS Kaydedildi: {ss_isim}')
                    global_alisveris['best_prio'] = prio
                    global_alisveris['event_name'] = e_name
                    global_alisveris['event_pid'] = pid
                    global_alisveris['ss_guncelle'] = True
                    global_alisveris['start_time'] = current_time
                else:
                    if prio <= global_alisveris['best_prio']:
                        global_alisveris['best_prio'] = prio
                        global_alisveris['event_name'] = e_name
                        global_alisveris['event_pid'] = pid
                        global_alisveris['ss_guncelle'] = True
                
                global_alisveris['last_event_time'] = current_time
            px1, py1, px2, py2 = map(int, box)
            px1, py1 = max(0, px1), max(0, py1)
            px2, py2 = min(W, px2), min(H, py2)
            h_box = py2 - py1
            w_box = px2 - px1
            ust_atla   = int(h_box * 0.30)
            sol_atla   = int(w_box * 0.10)
            sag_atla   = int(w_box * 0.05)
            person_roi = frame[py1 + ust_atla:py2, px1 + sol_atla:px2 - sag_atla]
            
            oran_yellow = 0.0
            oran_blue = 0.0
            
            if person_roi.size > 0:
                hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
                lower_yellow = np.array([20, 110, 80])
                upper_yellow = np.array([40, 255, 255])
                mask_yellow = cv2.inRange(hsv_roi, lower_yellow, upper_yellow)
                lower_yellow_bright = np.array([20, 110, 150])
                upper_yellow_bright = np.array([40, 255, 255])
                mask_yellow_bright = cv2.inRange(hsv_roi, lower_yellow_bright, upper_yellow_bright)
                lower_blue = np.array([95, 100, 80])
                upper_blue = np.array([115, 255, 255])
                mask_blue = cv2.inRange(hsv_roi, lower_blue, upper_blue)
                mask_blue_saturated = cv2.inRange(hsv_roi, np.array([95, 160, 80]), np.array([115, 255, 255]))
                lower_grey = np.array([0, 10, 70])
                upper_grey = np.array([180, 50, 150])
                mask_grey = cv2.inRange(hsv_roi, lower_grey, upper_grey)
                
                yellow_pixels = cv2.countNonZero(mask_yellow)
                yellow_bright_pixels = cv2.countNonZero(mask_yellow_bright)
                blue_pixels = cv2.countNonZero(mask_blue)
                blue_saturated_pixels = cv2.countNonZero(mask_blue_saturated)
                grey_pixels = cv2.countNonZero(mask_grey)
                total_pixels = person_roi.shape[0] * person_roi.shape[1]
                
                oran_yellow = yellow_pixels / total_pixels if total_pixels > 0 else 0
                oran_yellow_bright = yellow_bright_pixels / total_pixels if total_pixels > 0 else 0
                oran_blue = blue_pixels / total_pixels if total_pixels > 0 else 0
                oran_blue_saturated = blue_saturated_pixels / total_pixels if total_pixels > 0 else 0
                oran_grey = grey_pixels / total_pixels if total_pixels > 0 else 0
            
            if not kisiler_takip[pid]['personel_mi'] and person_roi.size > 0:
                doygunluk_orani = (oran_blue_saturated / oran_blue) if oran_blue > 0 else 0
                blue_sart = (oran_blue >= 0.015 and doygunluk_orani >= 0.60)
                kural_sari = oran_yellow >= 0.08 and oran_yellow_bright >= 0.015 and (blue_sart or oran_grey >= 0.15)
                kural_mavi = oran_blue >= 0.035 and doygunluk_orani >= 0.60 and oran_yellow >= 0.005
                
                if kural_sari or kural_mavi:
                    kisiler_takip[pid]['personel_skoru'] = kisiler_takip[pid].get('personel_skoru', 0) + 1
                    if kisiler_takip[pid]['personel_skoru'] >= 3:
                        kisiler_takip[pid]['personel_mi'] = True
            kisiler_takip[pid]['aktif_sure'] = kisiler_takip[pid].get('aktif_sure', 0) + fps_aralik
            gecen_sure = kisiler_takip[pid]['aktif_sure']
            
            if kisiler_takip[pid]['personel_mi']:
                kisi_renk = (0, 165, 255)
                durum_etiketi = f"OPET PERSONEL (S:{oran_yellow*100:.1f}% M:{oran_blue*100:.1f}%)"
            else:
                if gecen_sure >= 10.0 and not kisiler_takip[pid]['sayildi']:
                    kisiler_takip[pid]['sayildi'] = True
                    kisi_sayisi += 1
                    son_olay_zaman = current_time
                    log('TAKIP', f'{kisi_sayisi}. Kisi Sayildi (Kisi ID: {pid} alanda 10s kaldi)')
                    islem_kaydet(4, "Kisi_Takip")
                kisi_renk = (255, 0, 0) if kisiler_takip[pid]['sayildi'] else (255, 0, 255)
                durum_etiketi = f"KISI ID: {pid} ({int(gecen_sure)}s)"

            cv2.rectangle(frame, (px1, py1), (px2, py2), kisi_renk, 2)
            cv2.putText(frame, durum_etiketi, (px1, py1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, kisi_renk, 2)
            if len(keypoints_data) > i:
                person_kpts = keypoints_data[i]
                for wrist_idx in [9, 10]:
                    if confs is not None and confs[i][wrist_idx] < 0.3:
                        continue
                        
                    wx, wy = int(person_kpts[wrist_idx][0]), int(person_kpts[wrist_idx][1])
                    if wx == 0 and wy == 0:
                        continue
                    cv2.circle(frame, (wx, wy), 8, (0, 255, 0), -1)
                    cv2.putText(frame, 'El/Bilek', (wx+10, wy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    if gecen_sure > 1.0:
                        is_personel_now = kisiler_takip[pid]['personel_mi']
                        if FINCAN_X1 <= wx <= FINCAN_X2 and FINCAN_Y1 <= wy <= FINCAN_Y2:
                            nesne_bolgede = True
                            if not is_personel_now:
                                nesne_bolgede_musteri = True
                                islem_kaydet(3, "Teslim")
                        if FIS_X1 <= wx <= FIS_X2 and FIS_Y1 <= wy <= FIS_Y2:
                            el_fiste = True
                            if not is_personel_now:
                                islem_kaydet(1, "Fis_Okuyucu")
                        if FIS_X1 <= wx <= (FIS_X2 + 300) and FIS_Y1 <= wy <= (FIS_Y2 + 300):
                            el_fis_genis_alanda = True
                            if beyaz_piksel_sayisi > min_giris_esigi:
                                if not is_personel_now:
                                    islem_kaydet(1, "Fis_Okuyucu")
                        if EL_X1 <= wx <= EL_X2 and EL_Y1 <= wy <= EL_Y2:
                            bilek_bolgede = True
                            if not is_personel_now:
                                islem_kaydet(2, "Fis_Alim")
    if nesne_bolgede:
        son_kahve_zaman = current_time
        son_olay_zaman = current_time
        if not islem_aktif:
            if ilk_kare_okundu:
                islem_aktif = True
                if nesne_bolgede_musteri:
                    kahve_sayisi += 1
                    satis_kaydet(kahve_sayisi)
                    log('ISLEM', f'{kahve_sayisi}. Kahve - Musteri el bolgeye girdi.')
                else:
                    log('ISLEM', 'Personel kahve bolgesine mudehale etti (Satis yazilmayacak, sadece Video kaydedilecek).')
    elif islem_aktif and (current_time - son_kahve_zaman) > 10.0:
        islem_aktif = False
        log('ISLEM', f'Kahve islemi bitti (10 sn gecti).')
    kutu_renk = (0, 255, 0) if nesne_bolgede else (0, 165, 255)
    cv2.rectangle(frame, (FINCAN_X1, FINCAN_Y1), (FINCAN_X2, FINCAN_Y2), kutu_renk, 2)
    cv2.putText(frame, 'Teslim/ Bilek', (FINCAN_X1, FINCAN_Y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, kutu_renk, 2)
    cv2.rectangle(frame, (KISI_X1, KISI_Y1), (KISI_X2, KISI_Y2), (255, 0, 255), 2)
    cv2.putText(frame, 'Kisi Takip Alani', (KISI_X1, KISI_Y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    if bilek_bolgede:
        son_bilek_zaman = current_time
        son_olay_zaman = current_time
        if not bilek_aktif:
            if ilk_kare_okundu:
                bilek_aktif = True
                bilek_sayisi += 1
                log('BILEK', f'{bilek_sayisi}. Bilek Tespiti Yapildi')
    elif bilek_aktif and (current_time - son_bilek_zaman) > 10.0:
        bilek_aktif = False
    bilek_renk = (0, 255, 0) if bilek_bolgede else (0, 165, 255)
    cv2.rectangle(frame, (EL_X1, EL_Y1), (EL_X2, EL_Y2), bilek_renk, 2)
    cv2.putText(frame, 'Fis Alim/ Bilek', (EL_X1, EL_Y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, bilek_renk, 2)
    fis_okutuldu_mu = False
    if FIS_X2 > FIS_X1 and FIS_Y2 > FIS_Y1:
        fis_gecerli = False
        if el_fiste:
            fis_gecerli = True
        elif (beyaz_piksel_sayisi > min_giris_esigi) and el_fis_genis_alanda:
            fis_gecerli = True
        
        if fis_gecerli:
            son_fis_zaman = current_time
            son_olay_zaman = current_time
            if not fis_aktif:
                if ilk_kare_okundu:
                    fis_aktif = True
                    fis_sayisi += 1
                    log('FIS', f'{fis_sayisi}. Fis Okutuldu (Beyaz/YOLO26)')
        elif fis_aktif and (current_time - son_fis_zaman) > 10.0:
            fis_aktif = False
            
        ilk_kare_okundu = True
    fis_renk = (0, 0, 255) if fis_gecerli else (255, 255, 0)
    cv2.rectangle(frame, (FIS_X1, FIS_Y1), (FIS_X2, FIS_Y2), fis_renk, 2)
    cv2.putText(frame, 'Fis Okuyucu/ White', (FIS_X1, FIS_Y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, fis_renk, 2)
    cv2.rectangle(frame, (FIS_X1, FIS_Y1), (FIS_X2 + 300, FIS_Y2 + 300), (100, 100, 100), 1)
    if islem_aktif:
        son_olay_zaman = current_time
    if (nesne_bolgede) or (current_time - son_olay_zaman < 2.0):
        if not kayit_aktif:
            kayit_aktif = True
            zaman_damga = time.strftime('%Y-%m-%d_%H-%M-%S')
            kayit_dosyasi = os.path.join(KAYIT_KLASORU, f'olay_{zaman_damga}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_writer = cv2.VideoWriter(kayit_dosyasi, fourcc, FPS_SINIRI, (W, H))
            
            for onceki_kare in on_buffer:
                video_writer.write(onceki_kare)
            log('KAYIT', f'Video kaydi basladi (Geçmişle Birlikte): {kayit_dosyasi}')
            
    else:
        if kayit_aktif and (current_time - son_olay_zaman) > BOSLUK_SURESI:
            kayit_aktif = False
            if video_writer is not None:
                video_writer.release()
                video_writer = None
            log('KAYIT', f'Video kaydi tamamlandi ve kaydedildi.')
    durum_renk = (0, 0, 255) if islem_aktif else (0, 255, 0)
    durum_metin = 'KAYIT AKTIF' if islem_aktif else 'BEKLIYOR...'
    sayici_renk = (0, 255, 0) if islem_aktif else (0, 0, 255)
    
    cv2.putText(frame, f'Teslim/ Bilek: {kahve_sayisi}', (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, sayici_renk, 3)
    cv2.putText(frame, f'Kisi Sayimi: {kisi_sayisi}', (30, 195), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    fis_yazi_renk = (0, 255, 0) if fis_gecerli else (0, 0, 255)
    cv2.putText(frame, f'Fis Okuyucu/ White: {fis_sayisi}', (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.2, fis_yazi_renk, 3)
    bilek_yazi_renk = (0, 255, 0) if bilek_bolgede else (0, 0, 255)
    cv2.putText(frame, f'Fis Alim/ Bilek: {bilek_sayisi}', (30, 285), cv2.FONT_HERSHEY_SIMPLEX, 1.2, bilek_yazi_renk, 3)
    
    cv2.putText(frame, durum_metin, (30, 330), cv2.FONT_HERSHEY_SIMPLEX, 1.0, durum_renk, 2)
    gecen_bosluk = current_time - global_alisveris['last_event_time']
    gecen_toplam = current_time - global_alisveris['start_time']
    if global_alisveris['best_frame'] is not None and (gecen_bosluk > 10.0 or gecen_toplam > 35.0):
        ss_isim = f"Screenshot_Prio{global_alisveris['best_prio']}_{global_alisveris['event_name']}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        ss_path = os.path.join(SCREENSHOT_KLASORU, ss_isim)
        cv2.imwrite(ss_path, global_alisveris['best_frame'])
        sebep = "10s bosluk" if gecen_bosluk > 10.0 else "35s siniri"
        log('SCREENSHOT', f'Alisveris tamamlandi ({sebep}). SS Kaydedildi: {ss_isim}')
        global_alisveris['best_frame'] = None
    if global_alisveris['ss_guncelle']:
        global_alisveris['best_frame'] = frame.copy()
        global_alisveris['ss_guncelle'] = False
    on_buffer.append(frame.copy())

    if kayit_aktif and video_writer is not None:
        video_writer.write(frame)

    try:
        _, buffer = cv2.imencode('.jpg', frame)
        current_frame_jpeg = buffer.tobytes()
    except Exception:
        pass

if video_writer is not None:
    video_writer.release()

cap.release()
try:
    cv2.destroyAllWindows()
except Exception:
    pass
