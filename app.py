import json
import threading
import time
import os
import logging
import numpy as np
from urllib.request import urlopen, Request
from flask import Flask, jsonify
from collections import defaultdict, Counter, deque

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

HOST = '0.0.0.0'
POLL_INTERVAL = 5
RETRY_DELAY = 5
MAX_HISTORY = 50

lock_100 = threading.Lock()
lock_101 = threading.Lock()

latest_result_100 = {
    "Phien_hien_tai": 0, "Xuc_xac": 0, "Tong": 0, "Ket_qua": "Chưa có",
    "Phien_tiep_theo": 0, "Du_doan": "Chưa có", "Do_tin_cay": 0.0,
    "Giai_thich": "Chưa có dữ liệu", "id": "Vannhat_Comeback"
}
latest_result_101 = {
    "Phien_hien_tai": 0, "Xuc_xac": 0, "Tong": 0, "Ket_qua": "Chưa có",
    "Phien_tiep_theo": 0, "Du_doan": "Chưa có", "Do_tin_cay": 0.0,
    "Giai_thich": "Chưa có dữ liệu", "id": "Vannhat_Comeback"
}

history_100 = deque(maxlen=MAX_HISTORY)
history_101 = deque(maxlen=MAX_HISTORY)

last_sid_100 = None
last_sid_101 = None
sid_for_tx = None

def get_tai_xiu(d1, d2, d3):
    total = d1 + d2 + d3
    return "Xỉu" if total <= 10 else "Tài"

def analyze_patterns(ket_qua_history):
    """
    Phân tích mẫu cầu chuyên sâu, tối ưu hóa:
    - Tần suất, chuỗi bệt, đảo chiều (1-1, 2-2, 3-3).
    - Mẫu lặp (patterns như Tài-Xỉu lặp).
    - Độ biến động (std của chuỗi).
    - Giải thích dễ hiểu cho người chơi.
    """
    if len(ket_qua_history) < 2:
        return "Không đủ dữ liệu. Cầu cơ bản: 50% Tài/Xỉu. Theo dõi thêm để phát hiện bệt."

    n = len(ket_qua_history)
    recent = ket_qua_history[-min(20, n):]  # Phân tích 20 phiên gần nhất

    # Tần suất
    freq = Counter(recent)
    tai_freq = freq['Tài'] / len(recent) * 100
    xiu_freq = freq['Xỉu'] / len(recent) * 100

    # Chuỗi bệt
    streaks = []
    current_streak = 1
    current_type = recent[0]
    for i in range(1, len(recent)):
        if recent[i] == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_type = recent[i]
            current_streak = 1
    streaks.append((current_type, current_streak))
    current_streak_len = streaks[-1][1]
    avg_streak = np.mean([s[1] for s in streaks]) if streaks else 1
    streak_var = np.std([s[1] for s in streaks]) if len(streaks) > 1 else 0

    # Đảo chiều
    reversals_1 = sum(1 for i in range(1, len(recent)-1) if recent[i] != recent[i-1] and recent[i-1] == recent[i-2])
    reversal_rate_1 = (reversals_1 / (len(recent) - 2)) * 100 if len(recent) > 2 else 0
    reversals_2 = sum(1 for i in range(2, len(recent)) if recent[i] == recent[i-2] and recent[i-1] != recent[i])
    reversal_rate_2 = (reversals_2 / (len(recent) - 2)) * 100 if len(recent) > 2 else 0

    # Mẫu lặp đơn giản (Tài-Xỉu lặp)
    patterns = Counter()
    for i in range(2, len(recent)):
        pat = ''.join(recent[i-2:i])  # 2-step patterns
        patterns[pat] += 1
    top_pattern = patterns.most_common(1)[0] if patterns else ('', 0)

    # Giải thích
    analysis = f"Tần suất gần đây: Tài {tai_freq:.1f}%, Xỉu {xiu_freq:.1f}%. "
    analysis += f"Cầu bệt hiện tại: {current_streak_len} lần {current_type} (trung bình {avg_streak:.1f}, biến động {streak_var:.1f}). "
    analysis += f"Tỷ lệ đảo 1-1: {reversal_rate_1:.1f}%, 2-2: {reversal_rate_2:.1f}%. "
    if top_pattern[1] > 1:
        analysis += f"Mẫu lặp phổ biến: {top_pattern[0]} ({top_pattern[1]} lần). "
    analysis += f"Xu hướng: Nếu bệt > {avg_streak + streak_var:.0f}, dễ đảo chiều. Dễ chan theo Markov: Theo dõi chuyển tiếp từ {current_type}."

    return analysis

def predict_next(ket_qua_history):
    """
    Markov Chain bậc 1 tối ưu: Ma trận chuyển tiếp + điều chỉnh bệt.
    Độ tin cậy: Dựa trên max_prob và ổn định (1 - std freq).
    Không random: Toàn bộ dựa trên dữ liệu.
    """
    if not ket_qua_history:
        return "Tài", 15.0, "Dự đoán khởi tạo: Cân bằng lý thuyết."

    states = ['Tài', 'Xỉu']
    trans_count = defaultdict(lambda: defaultdict(int))

    # Xây ma trận từ lịch sử
    for i in range(1, len(ket_qua_history)):
        prev, curr = ket_qua_history[i-1], ket_qua_history[i]
        trans_count[prev][curr] += 1

    trans_prob = {}
    for prev in states:
        total_prev = sum(trans_count[prev].values())
        if total_prev > 0:
            trans_prob[prev] = {next_s: count / total_prev for next_s, count in trans_count[prev].items()}
        else:
            # Fallback tần suất toàn cục
            total = len(ket_qua_history)
            freq = Counter(ket_qua_history)
            trans_prob[prev] = {'Tài': freq['Tài'] / total if total > 0 else 0.5,
                                'Xỉu': freq['Xỉu'] / total if total > 0 else 0.5}

    last_state = ket_qua_history[0]
    probs = trans_prob[last_state]
    predicted = max(probs, key=probs.get)
    max_prob = probs[predicted]

    # Điều chỉnh bệt (từ analyze_patterns)
    current_streak = 1
    for i in range(1, len(ket_qua_history)):
        if ket_qua_history[i] == last_state:
            current_streak += 1
        else:
            break
    if current_streak >= 3:
        opposite = 'Xỉu' if predicted == 'Tài' else 'Tài'
        reversal_boost = min(current_streak * 0.1, 0.2)
        probs[opposite] += reversal_boost
        probs[predicted] -= reversal_boost
        predicted = max(probs, key=probs.get)
        max_prob = probs[predicted]

    # Độ tin cậy
    if len(ket_qua_history) >= 10:
        window_freqs = [Counter(ket_qua_history[-w:])['Tài'] / w for w in [5, 10]]
        stability = 1 - np.std(window_freqs) if len(window_freqs) > 1 else 0.5
        confidence = max_prob * (50 + stability * 50)
    else:
        confidence = max_prob * 35 + 15

    explanation = analyze_patterns(ket_qua_history)
    explanation += f" Markov Chain dự đoán {predicted} (xác suất {max_prob*100:.1f}%, điều chỉnh bệt). Độ tin cậy dựa trên ổn định lịch sử."

    return predicted, round(confidence, 2), explanation

def update_result(store, history, lock, result, is_md5):
    with lock:
        base_result = {
            "Phien_hien_tai": result["Phien"],
            "Xuc_xac": result["Tong"],
            "Tong": result["Tong"],
            "Ket_qua": result["Ket_qua"],
            "Phien_tiep_theo": result["Phien"] + 1,
            "id": "Vannhat_Comeback"
        }
        store.clear()
        store.update(base_result)

        ket_qua_list = [h["Ket_qua"] for h in history if h["Ket_qua"] != "Chưa có"]
        du_doan, do_tin_cay, giai_thich = predict_next(ket_qua_list)
        store["Du_doan"] = du_doan
        store["Do_tin_cay"] = do_tin_cay
        store["Giai_thich"] = giai_thich

        full_result = base_result.copy()
        full_result.update({"Du_doan": du_doan, "Do_tin_cay": do_tin_cay, "Giai_thich": giai_thich})
        history.appendleft(full_result)

def poll_api(gid, lock, result_store, history, is_md5):
    global last_sid_100, last_sid_101, sid_for_tx
    url = f"https://jakpotgwab.geightdors.net/glms/v1/notify/taixiu?platform_id=g8&gid={gid}"
    while True:
        try:
            req = Request(url, headers={'User-Agent': 'Python-Proxy/1.0'})
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode('utf-8'))
            if data.get('status') == 'OK' and isinstance(data.get('data'), list):
                for game in data['data']:
                    cmd = game.get("cmd")
                    if not is_md5 and cmd == 1008:
                        sid_for_tx = game.get("sid")
                for game in data['data']:
                    cmd = game.get("cmd")
                    if is_md5 and cmd == 2006:
                        sid = game.get("sid")
                        d1, d2, d3 = game.get("d1"), game.get("d2"), game.get("d3")
                        if sid and sid != last_sid_101 and None not in (d1, d2, d3):
                            last_sid_101 = sid
                            total = d1 + d2 + d3
                            ket_qua = get_tai_xiu(d1, d2, d3)
                            result = {"Phien": sid, "Tong": total, "Ket_qua": ket_qua}
                            update_result(result_store, history, lock, result, is_md5)
                            logger.info(f"[MD5] Phiên {sid} - Markov dự đoán dựa trên {len(history)} phiên.")
                    elif not is_md5 and cmd == 1003:
                        d1, d2, d3 = game.get("d1"), game.get("d2"), game.get("d3")
                        sid = sid_for_tx
                        if sid and sid != last_sid_100 and None not in (d1, d2, d3):
                            last_sid_100 = sid
                            total = d1 + d2 + d3
                            ket_qua = get_tai_xiu(d1, d2, d3)
                            result = {"Phien": sid, "Tong": total, "Ket_qua": ket_qua}
                            update_result(result_store, history, lock, result, is_md5)
                            logger.info(f"[TX] Phiên {sid} - Markov dự đoán dựa trên {len(history)} phiên.")
                            sid_for_tx = None
        except Exception as e:
            logger.error(f"Lỗi API {gid}: {e}")
            time.sleep(RETRY_DELAY)
        time.sleep(POLL_INTERVAL)

app = Flask(__name__)

@app.route("/api/taixiu", methods=["GET"])
def get_taixiu_100():
    with lock_100:
        return jsonify(latest_result_100)

@app.route("/api/taixiumd5", methods=["GET"])
def get_taixiu_101():
    with lock_101:
        return jsonify(latest_result_101)

@app.route("/api/history", methods=["GET"])
def get_history():
    with lock_100, lock_101:
        return jsonify({"taixiu": list(history_100), "taixiumd5": list(history_101)})

@app.route("/")
def index():
    return "API Tài Xỉu VIP - Markov Chain Powered. Endpoints: /api/taixiu, /api/taixiumd5, /api/history"

if __name__ == "__main__":
    logger.info("Khởi động API Tài Xỉu VIP với Markov Chain...")
    thread_100 = threading.Thread(target=poll_api, args=("vgmn_100", lock_100, latest_result_100, history_100, False), daemon=True)
    thread_101 = threading.Thread(target=poll_api, args=("vgmn_101", lock_101, latest_result_101, history_101, True), daemon=True)
    thread_100.start()
    thread_101.start()
    logger.info("Polling bắt đầu - Phân tích mẫu cầu tối ưu.")
    port = int(os.environ.get("PORT", 8000))
    app.run(host=HOST, port=port)
