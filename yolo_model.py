import os
import shutil
import json
import re
from ultralytics import YOLO
import numpy as np
import cv2



def solve(image):
    model_path = "best (1).pt"
    model = YOLO(model_path)
    ## data moi
    test_dir = "./image"
    results_folder = "./results_test"
    os.makedirs(results_folder, exist_ok = True)


    img_results = "results_test/images_draw"
    label_results = "results_test/labels_predict"
    for path in [test_dir ,results_folder, img_results, img_results, label_results]:
        os.makedirs(path, exist_ok=True)
    os.makedirs(img_results, exist_ok=True)
    os.makedirs(label_results, exist_ok=True)
    image_path = os.path.join(test_dir, "ok.jpg")
    cv2.imwrite(image_path, image)
    for image in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image)

        results = model(image_path, conf= 0.1)

        # img_file = results[0].plot()
        output_image_path = os.path.join(img_results, image)
        results[0].save(output_image_path, labels=False)

        label_file = os.path.join(label_results, image.replace(".jpg", ".txt"))
        with open(label_file, "w") as f:
            for box in results[0].boxes:
                cls = int(box.cls.cpu().numpy())  # class ID: 0/1
                x, y, width, height = box.xywh[0].cpu().numpy()
                # x1, x2, y1, y2, = box.xxyy.cpu().numpy()
                f.write(f"{cls} {x:.2f} {y:.2f} {width:.2f} {height:.2f}\n")




    def getBoundboxes(img_path, label_path):
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        height, width = image.shape
        bounding_boxes = []
        with open(label_path, "r") as f:
            lines = f.readlines()
        circle = ""
        n = 0
        for i, line in enumerate(lines):
            data = line.strip().split()
            n = n + 1
            x, y, w, h = map(float, data[1:])
            x /= width
            y /= height
            w /= width
            h /= height

            bounding_boxes.append([x, y, w, h])
        bounding_boxes.sort(key=lambda x: x[1])
        SBD_MST_boxes = bounding_boxes[:9]
        SBD_MST_boxes.sort(key=lambda x: x[0])

        SBD_boxes = SBD_MST_boxes[:6]
        MDT_boxes = SBD_MST_boxes[6:]
        PartI_boxes = bounding_boxes[9: 49]
        PartII_boxes = bounding_boxes[49: 65]
        PartIII_boxes = bounding_boxes[65:]

        # Xử lí phần SBD_________________________________________________________________________________
        SBD_ans = []
        final_string = "SBD:"
        for box in SBD_boxes:
            x, y, w, h = box
            x *= width
            y *= height
            w *= width
            h *= height
            x, y, w, h = map(int, [x, y, w, h])
            row_img = image[y - h // 2: y + h // 2, x - w // 2: x + w // 2]

            kernel = np.ones((11, 11), np.uint8)

            row_img = cv2.morphologyEx(row_img, cv2.MORPH_CLOSE, kernel)

            row_img = cv2.GaussianBlur(row_img, (5, 5), 3)
            _, row_img = cv2.threshold(row_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # print(row_img.shape)
            x1 = x
            y1 = y - h // 2 + w // 2
            h1 = w1 = min(h, w)
            minGray = np.mean(row_img[:h1, :w1])
            for i in range(row_img.shape[0] - h1):
                # print(i)
                gray = np.mean(row_img[i:i + h1, :w1])
                if (gray < minGray):
                    minGray = gray
                    y1 = y - h // 2 + w // 2 + i
            SBD_mean = []
            for i in range(10):
                SBD_mean.append(np.mean(row_img[h * i // 10:(h * (i + 1)) // 10, :]))
            final_string += str(np.argmin(SBD_mean))
            # print(SBD_mean)
            # print('ok')
            # print(str(np.argmin(SBD_mean)))
            x1 /= width
            y1 /= height
            w1 /= width
            h1 /= height
            SBD_ans.append([x1, y1, w1, h1])

        # Xử lí phần MDT _______________________________________________________________________
        MDT_ans = []
        final_string += ",\nMDT:"
        for box in MDT_boxes:
            x, y, w, h = box
            x *= width
            y *= height
            w *= width
            h *= height
            x, y, w, h = map(int, [x, y, w, h])
            row_img = image[y - h // 2: y + h // 2, x - w // 2: x + w // 2]

            kernel = np.ones((11, 11), np.uint8)

            row_img = cv2.morphologyEx(row_img, cv2.MORPH_CLOSE, kernel)

            row_img = cv2.GaussianBlur(row_img, (5, 5), 3)
            _, row_img = cv2.threshold(row_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # print(row_img.shape)
            x1 = x
            y1 = y - h // 2 + w // 2
            h1 = w1 = min(h, w)
            minGray = np.mean(row_img[:h1, :w1])
            for i in range(row_img.shape[0] - h1):
                # print(i)
                gray = np.mean(row_img[i:i + h1, :w1])
                if (gray < minGray):
                    minGray = gray
                    y1 = y - h // 2 + w // 2 + i
            MDT_mean = []
            for i in range(10):
                MDT_mean.append(np.mean(row_img[h * i // 10:(h * (i + 1)) // 10, :]))
            final_string += str(np.argmin(MDT_mean))
            x1 /= width
            y1 /= height
            w1 /= width
            h1 /= height
            MDT_ans.append([x1, y1, w1, h1])

        # Xu ly part I _______________________________________________________________________________________
        PartI_ans = []
        PartI_boxes.sort(key=lambda x: x[0])
        PartIa = PartI_boxes[:10]
        PartIb = PartI_boxes[10:20]
        PartIc = PartI_boxes[20:30]
        PartId = PartI_boxes[30:]
        PartIa.sort(key=lambda x: x[1])
        PartIb.sort(key=lambda x: x[1])
        PartIc.sort(key=lambda x: x[1])
        PartId.sort(key=lambda x: x[1])
        PartI_boxes = np.concatenate((PartIa, PartIb, PartIc, PartId), axis=0)
        final_string += ",\nPart I:"
        for idd, box in enumerate(PartI_boxes):
            x, y, w, h = box
            x *= width
            y *= height
            w *= width
            h *= height
            x, y, w, h = map(int, [x, y, w, h])
            row_img = image[y - h // 2: y + h // 2, x - w // 2: x + w // 2]

            kernel = np.ones((11, 11), np.uint8)

            row_img = cv2.morphologyEx(row_img, cv2.MORPH_CLOSE, kernel)

            row_img = cv2.GaussianBlur(row_img, (5, 5), 3)
            _, row_img = cv2.threshold(row_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # print(row_img.shape)
            # plt.figure(figsize=(30, 30))
            # plt.imshow(row_img, cmap='gray'), plt.axis("off")
            # plt.tight_layout()
            # plt.show()
            x1 = x - w // 2 + h // 2
            y1 = y
            h1 = w1 = min(h, w)
            minGray = np.mean(row_img[:h1, :w1])
            for i in range(row_img.shape[1] - w1):
                # print(i)
                gray = np.mean(row_img[:h1, i:i + w1])
                if (gray < minGray):
                    minGray = gray
                    x1 = x - w // 2 + h // 2 + i
            mp = ["A", "B", "C", "D"]
            partI_mean = []
            for i in range(4):
                partI_mean.append(np.mean(row_img[:, w * i // 4:(w * (i + 1)) // 4]))
            final_string += str(idd + 1) + ": " + mp[(np.argmin(partI_mean))] + ",\n"
            x1 /= width
            y1 /= height
            w1 /= width
            h1 /= height

            PartI_ans.append([x1, y1, w1, h1])

        # Xu ly part II_______________________________________________________________________________________
        PartII_ans = []
        PartII_boxes.sort(key=lambda x: x[0])
        PartIIa = PartII_boxes[:4]
        PartIIb = PartII_boxes[4:8]
        PartIIc = PartII_boxes[8:12]
        PartIId = PartII_boxes[12:]
        PartIIa.sort(key=lambda x: x[1])
        PartIIb.sort(key=lambda x: x[1])
        PartIIc.sort(key=lambda x: x[1])
        PartIId.sort(key=lambda x: x[1])
        PartII_boxes = np.concatenate((PartIIa, PartIIb, PartIIc, PartIId), axis=0)
        PartII_filter_boxes = []
        for i in range(32):
            box = PartII_boxes[i // 8 * 4 + i % 4]
            if i % 8 < 4:
                PartII_filter_boxes.append([box[0] - box[2] / 4, box[1], box[2] / 2, box[3]])
            else:
                PartII_filter_boxes.append([box[0] + box[2] / 4, box[1], box[2] / 2, box[3]])

        final_string += ",\nPart II: "
        for idd, box in enumerate(PartII_filter_boxes):
            x, y, w, h = box
            x *= width
            y *= height
            w *= width
            h *= height
            x, y, w, h = map(int, [x, y, w, h])
            row_img = image[y - h // 2: y + h // 2, x - w // 2: x + w // 2]

            kernel = np.ones((11, 11), np.uint8)

            row_img = cv2.morphologyEx(row_img, cv2.MORPH_CLOSE, kernel)

            row_img = cv2.GaussianBlur(row_img, (5, 5), 3)
            _, row_img = cv2.threshold(row_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # print(row_img.shape)
            # plt.figure(figsize=(30, 30))
            # plt.imshow(row_img, cmap='gray'), plt.axis("off")
            # plt.tight_layout()
            # plt.show()
            x1 = x - w // 2 + h // 2
            y1 = y
            h1 = w1 = min(h, w)
            minGray = np.mean(row_img[:h1, :w1])
            for i in range(row_img.shape[1] - w1):
                # print(i)
                gray = np.mean(row_img[:h1, i:i + w1])
                if (gray < minGray):
                    minGray = gray
                    x1 = x - w // 2 + h // 2 + i
            mp = ["true", "false"]
            partII_mean = []
            for i in range(2):
                partII_mean.append(np.mean(row_img[:, w * i // 2:(w * (i + 1)) // 2]))
            final_string += str(idd) + ". " + mp[np.argmin(partII_mean)] + ",\n"
            x1 /= width
            y1 /= height
            w1 /= width
            h1 /= height
            # print([x1, y1, w1, h1])
            PartII_ans.append([x1, y1, w1, h1])

        # Xu ly part III_______________________________________________________________________________________
        PartIII_ans = []
        PartIII_boxes.sort(key=lambda x: x[0])
        final_string += ",\nPart III:"
        for idd, box in enumerate(PartIII_boxes):
            x, y, w, h = box
            x *= width
            y *= height
            w *= width
            h *= height
            x, y, w, h = map(int, [x, y, w, h])
            row_img = image[y - h // 2: y + h // 2, x - w // 2: x + w // 2]

            kernel = np.ones((11, 11), np.uint8)

            row_img = cv2.morphologyEx(row_img, cv2.MORPH_CLOSE, kernel)

            row_img = cv2.GaussianBlur(row_img, (5, 5), 3)
            _, row_img = cv2.threshold(row_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            # print(row_img.shape)
            # plt.figure(figsize=(30, 30))
            # plt.imshow(row_img, cmap='gray'), plt.axis("off")
            # plt.tight_layout()
            # plt.show()
            x1 = x
            y1 = y - h // 2 + w // 2
            h1 = w1 = min(h, w)
            minGray = np.mean(row_img[:h1, :w1])
            for i in range(row_img.shape[0] - h1):
                # print(i)
                gray = np.mean(row_img[i:i + h1, :w1])
                if (gray < minGray):
                    minGray = gray
                    y1 = y - h // 2 + w // 2 + i
            partIII_mean = []
            mp = ["-", ".", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

            if idd % 4 == 0:
                final_string += str(idd // 4 + 1) + ". "
                for i in range(12):
                    partIII_mean.append(np.mean(row_img[h * i // 12:(h * (i + 1)) // 12, :]))
                final_string += mp[np.argmin(partIII_mean)]
            elif idd % 4 == 1 or idd % 4 == 2:
                for i in range(11):
                    partIII_mean.append(np.mean(row_img[h * i // 11:(h * (i + 1)) // 11, :]))
                final_string += mp[np.argmin(partIII_mean) + 1]
            else:
                for i in range(10):
                    partIII_mean.append(np.mean(row_img[h * i // 10:(h * (i + 1)) // 10, :]))
                final_string += mp[np.argmin(partIII_mean) + 2] + ""

            if idd % 4 == 3:
                final_string += ",\n"
            x1 /= width
            y1 /= height
            w1 /= width
            h1 /= height
            PartIII_ans.append([x1, y1, w1, h1])
            # print([x1, y1, w1, h1])

        def parse_result_to_json(raw_text):
            result = {
                "SBD": {},
                "MDT": {},
                "Part1": {},
                "Part2": {},
                "Part3": {}
            }

            # Remove redundant commas and normalize input
            raw_text = raw_text.strip().replace('\n', '').replace(', ,', ',')

            # Extract sections
            sbd_match = re.search(r'SBD:(\d+)', raw_text)
            mdt_match = re.search(r'MDT:(\d+)', raw_text)
            part1_match = re.search(r'Part I:(.*?),\s*Part II:', raw_text)
            part2_match = re.search(r'Part II:(.*?),\s*Part III:', raw_text)
            part3_match = re.search(r'Part III:(.*)', raw_text)

            if sbd_match:
                sbd_str = sbd_match.group(1)
                result["SBD"] = sbd_str 
                # print('ok fine')
                # print(sbd_str)

            if mdt_match:
                mdt_str = mdt_match.group(1)
                result["MDT"] = mdt_str
                # print('ok fine')

            if part1_match:
                part1_entries = part1_match.group(1).split(',')
                for entry in part1_entries:
                    if ':' in entry:
                        q, ans = entry.strip().split(':')
                        result["Part1"][int(q.strip())] = ans.strip()

            if part2_match:
                part2_entries = [e.strip() for e in part2_match.group(1).split(',') if e.strip()]
                for entry in part2_entries:
                    match = re.match(r'(\d+)\.\s*(true|false)', entry.strip(), re.IGNORECASE)
                    if match:
                        q, val = match.groups()
                        result["Part2"][q] = val.capitalize()

            if part3_match:
                part3_lines = [line.strip() for line in part3_match.group(1).split(',') if line.strip()]
                for line in part3_lines:
                    match = re.match(r'(\d+)\.\s*(.+)', line)
                    if match:
                        q, val = match.groups()
                        result["Part3"][q] = val.replace(',', '.')

            return json.dumps(result, indent=2, ensure_ascii=False)

        from itertools import islice
        def json_to_expanded_dict(json_string):
            raw = json.loads(json_string)
            result = {}

            # SBD & MDT
            for key in ["SBD", "MDT"]:
                # print('yes')
                # print(raw[key])
                result[key] = raw[key]

            # Part1
            result["Part1"] = {int(k): v if v.strip() else [] for k, v in islice(raw["Part1"].items(), 12)}

            # Part2
            result["Part2"] = {int(k) + 1: str(v == "True") if v.strip() else [] for k, v in
                               islice(raw["Part2"].items(), 16)}

            # Step 1: Lấy dữ liệu Part3 ban đầu
            part3_raw = {int(k): v if v.strip() else [] for k, v in islice(raw["Part3"].items(), 24)}
            
            # Gán vào result
            result["Part3"] = part3_raw

            return result
        final_string = parse_result_to_json(final_string)
        final_string = json_to_expanded_dict(final_string)
        #final_string = json.dumps(final_string, indent=2, ensure_ascii=False)

        return SBD_ans, MDT_ans, PartI_ans, PartII_ans, PartIII_ans, final_string


    model_path = "best (1).pt"
    model = YOLO(model_path)
    #test_dir = "/kaggle/input/test-set2/testset1/testset1/images"
    results_folder = "results_test_after"


    img_results =f"{results_folder}/images_draw"
    label_results = f"{results_folder}/labels_predict"
    label_final_draw = f"{results_folder}/labels_final"
    label_final_txt = f"{results_folder}/labels_final_txt"
    for path in [img_results, label_results, label_final_draw, label_final_txt]:
        os.makedirs(path, exist_ok=True)



    def getAnswer(test_dir, label_results, results_folder, img_results, label_final_draw, label_final_txt):
        for image in os.listdir(test_dir):
            image_path = os.path.join(test_dir, image)

            results = model(image_path,
                            conf=0.1)  # results ở đây là một list, không phải object nên ta phải truy cập vào phần tử đầu tiên

            # img_file = results[0].plot()
            output_image_path = os.path.join(img_results, image)
            results[0].save(output_image_path, labels=False)

            label_file = os.path.join(label_results, image.replace(".jpg", ".txt"))
            with open(label_file, "w") as f:
                for box in results[0].boxes:
                    cls = int(box.cls.cpu().numpy())  # class ID: 0/1
                    x, y, width, height = box.xywh[0].cpu().numpy()
                    # x1, x2, y1, y2, = box.xxyy.cpu().numpy()
                    f.write(f"{cls} {x:.2f} {y:.2f} {width:.2f} {height:.2f}\n")
        image_path = test_dir
        labels_path = label_results
        ret = ""
        for string in os.listdir(image_path):
            img_pth = os.path.join(image_path, string)
            label_pth = os.path.join(labels_path, string.replace('.jpg', '.txt'))
            SBD_ans, MDT_ans, PartI_ans, PartII_ans, PartIII_ans, final_string = getBoundboxes(img_pth, label_pth)
            image = cv2.imread(img_pth, cv2.IMREAD_GRAYSCALE)
            shape = image.shape[1::-1]
            for box in SBD_ans:
                top_left = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2]) * shape
                top_left = tuple(map(round, top_left))
                bottom_right = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2]) * shape
                bottom_right = tuple(map(round, bottom_right))
                color = (0, 255, 0)
                thickness = 3
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
            for box in MDT_ans:
                top_left = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2]) * shape
                top_left = tuple(map(round, top_left))
                bottom_right = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2]) * shape
                bottom_right = tuple(map(round, bottom_right))
                color = (0, 255, 0)
                thickness = 3
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
            for box in PartI_ans[:12]:
                top_left = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2]) * shape
                top_left = tuple(map(round, top_left))
                bottom_right = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2]) * shape
                bottom_right = tuple(map(round, bottom_right))
                color = (0, 255, 0)
                thickness = 3
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
            for box in PartII_ans[:16]:
                top_left = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2]) * shape
                top_left = tuple(map(round, top_left))
                bottom_right = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2]) * shape
                bottom_right = tuple(map(round, bottom_right))
                color = (0, 255, 0)
                thickness = 3
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
            for box in PartIII_ans[:24]:
                top_left = np.array([box[0] - box[2] / 2, box[1] - box[3] / 2]) * shape
                top_left = tuple(map(round, top_left))
                bottom_right = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2]) * shape
                bottom_right = tuple(map(round, bottom_right))
                color = (0, 255, 0)
                thickness = 3
                cv2.rectangle(image, top_left, bottom_right, color, thickness)
            cv2.imwrite(os.path.join(label_final_draw, string), image)
            with open(os.path.join(label_final_txt, string.replace('.jpg', '.txt')), "w") as f:
                f.write(json.dumps(final_string))
            return image, final_string

    image, final_string = getAnswer(test_dir, label_results, results_folder, img_results, label_final_draw, label_final_txt)
    return image, final_string
if __name__=='__main__':
    image = cv2.imread("image/IMG_1581_iter_100.jpg")
    image, final_string = solve(image)
    print(final_string)
    cv2.imwrite("output.jpg", image)
    print(type(final_string))
