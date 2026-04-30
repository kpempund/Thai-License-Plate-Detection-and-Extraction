from ultralytics import YOLO
import cv2

def crop_license_plate(model_path, image_path):
    image = cv2.imread(image_path)

    model = YOLO(model_path)

    results = model(image, imgsz=960, conf=0.5)[0]
    best = results.boxes[results.boxes.conf.argmax()]

    x1, y1, x2, y2 = best.xyxy[0].cpu().numpy().astype(int)
    crop = image[int(y1):int(y2), int(x1):int(x2)]

    return crop

def extract_plate_text(model_path, lp):
    if lp is None:
        return {"number": "", "province": ""}

    model = YOLO(model_path)

    results = model(lp, verbose=False)[0]

    detected_chars = []
    detected_province = ""

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        x1 = float(box.xyxy[0][0])

        if cls_name in PROVINCE_MAP: # Province
            detected_province = PROVINCE_MAP[cls_name]
        elif cls_name in CHAR_DECODER: # Character
            detected_chars.append((x1, CHAR_DECODER[cls_name]))
        elif cls_name.isdigit(): # Number
            detected_chars.append((x1, cls_name))

    detected_chars.sort(key=lambda x: x[0])
    full_number = "".join([char for _, char in detected_chars])

    return {"number": full_number, "province": detected_province}

CHAR_DECODER = {
    "TH01": "ก", "TH02": "ข", "TH03": "ค", "TH04": "ฆ", "TH05": "ง",
    "TH06": "จ", "TH07": "ฉ", "TH08": "ช", "TH09": "ฌ", "TH10": "ญ",
    "TH11": "ฎ", "TH12": "ฐ", "TH13": "ฒ", "TH14": "ณ", "TH15": "ด",
    "TH16": "ต", "TH17": "ถ", "TH18": "ท", "TH19": "ธ", "TH20": "น",
    "TH21": "บ", "TH22": "ผ", "TH23": "พ", "TH24": "ฟ", "TH25": "ภ",
    "TH26": "ม", "TH27": "ย", "TH28": "ร", "TH29": "ล", "TH30": "ว",
    "TH31": "ศ", "TH32": "ษ", "TH33": "ส", "TH34": "ห", "TH35": "ฬ",
    "TH36": "อ", "TH37": "ฮ"
}

PROVINCE_MAP = {
    "ATG": "อ่างทอง", "AYA": "พระนครศรีอยุธยา", "BKK": "กรุงเทพมหานคร",
    "BKN": "บึงกาฬ", "BRM": "บุรีรัมย์", "CBI": "ชลบุรี",
    "CCO": "ฉะเชิงเทรา", "CMI": "เชียงใหม่", "CNT": "ชัยนาท",
    "CPM": "ชัยภูมิ", "CPN": "ชุมพร", "CRI": "เชียงราย",
    "CTI": "จันทบุรี", "KBI": "กระบี่", "KKN": "ขอนแก่น",
    "KPT": "กำแพงเพชร", "KRI": "กาญจนบุรี", "KSN": "กาฬสินธุ์",
    "LEI": "เลย", "LPG": "ลำปาง", "LPN": "ลำพูน",
    "LRI": "ลพบุรี", "MDH": "มุกดาหาร", "MKM": "มหาสารคาม",
    "NAN": "น่าน", "NBI": "นนทบุรี", "NBP": "หนองบัวลำภู",
    "NKI": "หนองคาย", "NMA": "นครราชสีมา", "NPM": "นครพนม",
    "NPT": "นครปฐม", "NSN": "นครสวรรค์", "NST": "นครศรีธรรมราช",
    "NYK": "นครนายก", "PBI": "เพชรบุรี", "PCT": "พิจิตร",
    "PKN": "ประจวบคีรีขันธ์", "PKT": "ภูเก็ต", "PLG": "พัทลุง",
    "PLK": "พิษณุโลก", "PNB": "พังงา", "PRE": "แพร่",
    "PRI": "ปราจีนบุรี", "PTE": "ปทุมธานี", "PYO": "พะเยา",
    "RBR": "ราชบุรี", "RET": "ร้อยเอ็ด", "RYG": "ระยอง",
    "SBR": "สระบุรี", "SKA": "สงขลา", "SKM": "สมุทรสงคราม",
    "SKN": "สมุทรสาคร", "SKW": "สระแก้ว", "SNI": "สุราษฎร์ธานี",
    "SNK": "สกลนคร", "SPB": "สุพรรณบุรี", "SPK": "สมุทรปราการ",
    "SRI": "สุรินทร์", "SRN": "สิงห์บุรี", "SSK": "ศรีสะเกษ",
    "STI": "สุโขทัย", "TAK": "ตาก", "TRT": "ตราด"
}

LP_DETECTOR_WEIGHTS = "models/lp_detect.pt"
LP_RECOGNIZER_WEIGHTS = "models/lp_recog.pt"

INPUT_IMAGE = "car2.jpg"

cropped_lp = crop_license_plate(model_path=LP_DETECTOR_WEIGHTS, image_path=INPUT_IMAGE)
pred = extract_plate_text(model_path=LP_RECOGNIZER_WEIGHTS, lp=cropped_lp)

pred_number = (pred.get("number") or "").strip()
pred_province = (pred.get("province") or "").strip()
print(pred_number+' '+pred_province)