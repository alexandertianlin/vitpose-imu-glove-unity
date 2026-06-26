import os, sys, json, math, time, socket, logging
import cv2, numpy as np, torch

os.environ["PYOPENGL_PLATFORM"] = "wgl"
os.environ["MMENGINE_LOGLEVEL"] = "ERROR"
logging.getLogger().setLevel(logging.ERROR)
np.bool = bool

HAMER_DIR = r"C:\Users\Administrator\Documents\Codex\2026-06-16\files-mentioned-by-the-user-gpu2-3\hamer_code\hamer-main"
CKPT = os.path.join(HAMER_DIR, "_DATA", "hamer_ckpts", "checkpoints", "hamer.ckpt")
os.chdir(HAMER_DIR)
sys.path.insert(0, HAMER_DIR)
sys.path.insert(0, os.path.join(HAMER_DIR, "third-party", "ViTPose"))
from hamer.models import load_hamer
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils import recursive_to
from vitpose_model import ViTPoseModel

g_kp3d = None; seq = 0; last_send = 0; SEND_INTV = 0.033
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_addr = ("127.0.0.1", 5055)

FN = ["thumb","index","middle","ring","little"]
FC = [{"mcp":2,"pip":3,"dip":3,"tip":4},{"mcp":5,"pip":6,"dip":7,"tip":8},{"mcp":9,"pip":10,"dip":11,"tip":12},{"mcp":13,"pip":14,"dip":15,"tip":16},{"mcp":17,"pip":18,"dip":19,"tip":20}]
W = 0; I0 = 5; M0 = 9; M3 = 12; L0 = 17

def nrm(v):
    n = np.linalg.norm(v); return v / n if n > 1e-8 else v

def orient_q_from_landmarks(pw):
    u = nrm(pw[W] - pw[M0]); r = nrm(pw[I0] - pw[L0])
    f = nrm(np.cross(r, u)); u = nrm(np.cross(f, r))
    R = np.column_stack([r, u, f])
    tr = R[0,0]+R[1,1]+R[2,2]
    if tr > 0:
        s = math.sqrt(tr+1.0)*2; q = [s*0.25, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
        q = [(R[2,1]-R[1,2])/s, s*0.25, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
        q = [(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, s*0.25, (R[1,2]+R[2,1])/s]
    else:
        s = math.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
        q = [(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, s*0.25]
    nv = math.sqrt(sum(x*x for x in q))
    return [float(x/nv) for x in q] if nv > 1e-10 else [1,0,0,0]

def curl_from_kp(pw, n):
    c = FC[n]; m = pw[c["mcp"]]; p = pw[c["pip"]]; d = pw[c["dip"]]; t = pw[c["tip"]]
    chord = np.linalg.norm(t - m)
    pip2tip = np.linalg.norm(t - p)
    if n == 0:
        arc = np.linalg.norm(m-p) + np.linalg.norm(p-t)
    else:
        arc = np.linalg.norm(m-p) + np.linalg.norm(p-d) + np.linalg.norm(d-t)
    if arc < 1e-8: return 0.0
    return 1.0 - max(0, min(1, 0.8*chord/arc + 0.2*pip2tip/arc))

def spread_from_kp(pw, n):
    if n == 2: return 0.0
    c = FC[n]; mc = FC[2]
    fd = nrm(pw[c["tip"]] - pw[c["mcp"]])
    md = nrm(pw[mc["tip"]] - pw[mc["mcp"]])
    ddot = max(-1, min(1, np.dot(fd, md)))
    return max(0, min(1, math.acos(ddot) / 1.2))

print("Loading HAMER ...", flush=True)
m_hamer, cfg_h = load_hamer(CKPT, init_renderer=False)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m_hamer = m_hamer.to(dev).eval()
print("Loading ViTPose Hand ...", flush=True)
vitpose = ViTPoseModel("cuda" if torch.cuda.is_available() else "cpu")
vitpose.set_model("ViTPose-base-hand")
print("ViTPose-base-hand ready", flush=True)

print("Opening D435i ...", flush=True)
import pyrealsense2 as rs
pipe = rs.pipeline()
cfg_rs = rs.config()
cfg_rs.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg_rs)
for _ in range(30): pipe.wait_for_frames()
print("D435i OK. Press q/ESC to quit.", flush=True)

fc = 0; ft = time.time(); fps = 0
last_boxes = None; last_right = None
hands_confirmed = True; consecutive_no_hand = 0
last_kp2d = None; last_hand_center = None

try:
    while True:
        frames = pipe.wait_for_frames()
        cf = frames.get_color_frame()
        if not cf: continue
        img = np.asanyarray(cf.get_data())
        h, w, _ = img.shape
        g_kp3d = None

        boxes_list = []; rights_list = []

        if last_boxes is not None:
            box = last_boxes[0]
        else:
            cx, cy = w // 2, h // 2
            bs = 200
            box = [cx - bs//2, cy - bs//2, cx + bs//2, cy + bs//2]
        det = [np.array([[box[0], box[1], box[2], box[3], 0.9]])]
        vitposes = vitpose.predict_pose(img, det)
        for vp in vitposes:
            kps = vp["keypoints"]
            v = kps[:,2] > 0.3
            if sum(v) < 5: continue
            boxes_list.append([int(kps[v,0].min())-15, int(kps[v,1].min())-15,
                               int(kps[v,0].max())+15, int(kps[v,1].max())+15])
            rights_list.append(1.0)
        if boxes_list:
            consecutive_no_hand = 0
            last_boxes = np.stack(boxes_list).astype(np.float32)
            last_right = np.stack(rights_list)
            hands_confirmed = True
        else:
            consecutive_no_hand += 1
            if consecutive_no_hand > 5:
                last_boxes = None; last_right = None
                hands_confirmed = False

        if hands_confirmed == False and consecutive_no_hand == 1:
            seq += 1
            pkt = {"type":"hamer_hand","seq":seq,"ts":int(time.time()*1000),"num_hands":0}
            try:
                sock.sendto(json.dumps(pkt, separators=(",",":")).encode(), udp_addr)
                print("LOST_HAND seq=" + str(seq), flush=True)
            except:
                pass

        if not boxes_list and last_kp2d is not None and hands_confirmed and last_boxes is not None:
            centroid = last_kp2d.mean(axis=0)
            bw = last_boxes[0][2] - last_boxes[0][0]
            bh = last_boxes[0][3] - last_boxes[0][1]
            last_boxes[0] = [centroid[0]-bw/2, centroid[1]-bh/2, centroid[0]+bw/2, centroid[1]+bh/2]
            last_hand_center = (int(centroid[0]), int(centroid[1]))

        canvas = img.copy()
        if last_boxes is not None and hands_confirmed:
            ds = ViTDetDataset(cfg_h, img, last_boxes, last_right, rescale_factor=2.0)
            loader = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
            for batch in loader:
                batch = recursive_to(batch, dev)
                with torch.no_grad():
                    out = m_hamer(batch)
                for n in range(batch["img"].shape[0]):
                    kp3d = out["pred_keypoints_3d"][n].cpu().numpy()
                    g_kp3d = kp3d
                    kn = out["pred_keypoints_2d"][n].cpu().numpy() + 0.5
                    ir_val = batch["right"][n].item()
                    if ir_val < 0.5: kn[:,0] = 1.0 - kn[:,0]
                    cx = batch["box_center"][n,0].item()
                    cy = batch["box_center"][n,1].item()
                    bs = batch["box_size"][n].item()
                    kp2d = np.zeros((21,2), dtype=int)
                    kp2d[:,0] = (cx - bs/2 + kn[:,0]*bs).astype(int)
                    kp2d[:,1] = (cy - bs/2 + kn[:,1]*bs).astype(int)
                    last_kp2d = np.column_stack([kp2d[:,0].astype(np.float32), kp2d[:,1].astype(np.float32)])
                    last_hand_center = (int(kp2d[:,0].mean()), int(kp2d[:,1].mean()))
                    for e in [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)]:
                        cv2.line(canvas, tuple(kp2d[e[0]]), tuple(kp2d[e[1]]), (0,255,255), 2)
                    for p in kp2d: cv2.circle(canvas, tuple(p), 4, (0,255,0), -1)
                    cv2.putText(canvas, "R" if ir_val > 0.5 else "L", (int(cx)-10,int(cy)+5), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,0,0), 2)

        if hands_confirmed and g_kp3d is not None and time.time() - last_send >= SEND_INTV:
            curld = [curl_from_kp(g_kp3d, i) for i in range(5)]
            spreadd = [spread_from_kp(g_kp3d, i) for i in range(5)]
            seq += 1
            q = orient_q_from_landmarks(g_kp3d)
            pkt = {"type":"hamer_hand","seq":seq,"ts":int(time.time()*1000),"num_hands":1,
                   "hand_0_label":"right","hand_0_conf":0.9,"hand_0_wrist":g_kp3d[0].tolist(),
                   "hand_0_kp3d":g_kp3d.flatten().tolist(),"hand_0_orient_q":q}
            for i in range(5):
                pkt["hand_0_curl_"+FN[i]] = round(curld[i],3)
                pkt["hand_0_spread_"+FN[i]] = round(spreadd[i],3)
            try:
                sock.sendto(json.dumps(pkt, separators=(",",":")).encode(), udp_addr)
                print("UDP seq=" + str(seq), flush=True)
            except:
                pass
            last_send = time.time()
            ctxt = " ".join([FN[i]+":"+str(round(curld[i],2)) for i in range(5)])
            cv2.putText(canvas, ctxt, (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,200,0), 1)

        fc += 1
        if time.time()-ft >= 1: fps,fc,ft = fc,0,time.time()
        cv2.putText(canvas, "FPS:{} UDP:5055".format(fps), (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,0), 1)
        cv2.imshow("ViTPose Hand -> Unity 5055", canvas)
        k = cv2.waitKey(1)&0xFF
        if k in (ord('q'),27): break
except Exception as e:
    print("Error: {}".format(e))
finally:
    pipe.stop(); cv2.destroyAllWindows(); print("Done.")
