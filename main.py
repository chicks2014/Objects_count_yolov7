import glob
import shutil
from fastapi import FastAPI
import os
import random as rd
import base64
import subprocess
from subprocess import Popen, PIPE
import cv2 as cv
import numpy as np
from imutils.object_detection import non_max_suppression

from pydantic import BaseModel

app = FastAPI()

inference_path = "inference/images"
run_path = "runs/detect/exp"


class Item(BaseModel):
    categoryName: str
    imageSource: str
    selectedPortion: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


def image_match_count(file_name, template_name):
    src_img = cv.imread(f"./{inference_path}/{file_name}")  # main img
    template = cv.imread(f"./{inference_path}/{template_name}")  # temp img

    print("--" * 50)
    print(f"src_img: {file_name} - \n templ: {template_name}")
    print("--" * 50)

    img_gray = cv.cvtColor(src_img, cv.COLOR_BGR2GRAY)
    template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    (tH, tW) = template.shape[:2]
    # print("src_img shape: ", src_img.shape[:2])
    (tH_src, tW_src) = template.shape[:2]

    result = cv.matchTemplate(img_gray, template_gray, cv.TM_CCOEFF_NORMED)

    (yCoords, xCoords) = np.where(result >= 0.5)
    clone = src_img.copy()

    rects = []
    # loop over the starting (x, y)-coordinates again
    for (x, y) in zip(xCoords, yCoords):
        # update our list of rectangles
        rects.append((x, y, x + tW, y + tH))
    # apply non-maxima suppression to the rectangles
    pick = non_max_suppression(np.array(rects))

    print(f"Total count = {len(pick)}")

    print("[INFO] {} matched locations *after* NMS".format(len(pick)))
    # loop over the final bounding boxes
    for (startX, startY, endX, endY) in pick:
        # draw the bounding box on the image
        cv.rectangle(src_img, (startX, startY), (endX, endY), (255, 0, 0), 3)

    # font
    font = cv.FONT_HERSHEY_SIMPLEX

    # org
    # org = (tH_src - 100, tW_src - 100)
    org = (int(tH_src - int(tH_src) * 0.10), int(tW_src - int(tW_src) * 0.10))

    # bottom_right = (tH_src-10, tW_src-10)

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (0, 0, 255)

    # Line thickness of 2 px
    thickness = 2
    obj_cnts = len(pick)
    cv.putText(
        src_img,
        f"counts={obj_cnts}",
        org,
        font,
        fontScale,
        color,
        thickness,
        cv.LINE_AA,
    )
    # show the output image
    dest_path = f"./{run_path}/{file_name}"
    print(f"dest_path: {dest_path}")
    # cv.imwrite("./imgs/ind_coins100_final.png", src_img)
    cv.imwrite(dest_path, src_img)

    with open(os.path.join(run_path, file_name), "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return {
            "status": obj_cnts > 0,
            "result": encoded_string if obj_cnts > 0 else None,
            "count": obj_cnts,
        }


def generate_inferences(file_name, category_name):
    cnt = 0
    # generate inferences
    result = subprocess.run(
        f"python ./yolov7/detect_and_count.py --weights ./yolov7/mybest_40.pt --conf 0.1 --classes {category_name} --source ./{inference_path}/{file_name}",
        capture_output=True,
        text=True,
        shell=True,
        check=True,
    )

    #  result = subprocess.run(
    #     f'python ./yolov7/detect_and_count.py --weights ./yolov7/mybest_40.pt --conf 0.1 --classes {category_name} --source ./inference/images/{file_name}',
    #     capture_output=True, shell=True, check=True)

    # result = subprocess.run(['python' ,'./yolov7/detect_and_count.py', '--weights', './yolov7/mybest_40.pt', '--conf' ,'0.1', '--source', f'./inference/images/{file_name}'],
    #     shell=True, stdin=PIPE, stdout=PIPE)
    # result.communicate()
    # result.stdout.close()
    # out, err = result.communicate()
    print("\n\n result.stdout -- \n\n")
    print(result.stdout)
    # print(err)
    for l in result.stdout.splitlines():
        print(type(l))
        if "total_counts" in l:
            cnt = l.split("=")[1]
        print(l)

    with open(os.path.join(run_path, file_name), "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return {
            "status": int(cnt) > 0,
            "result": encoded_string if int(cnt) > 0 else None,
            "count": int(cnt),
        }


def clear_dir(path):
    files = glob.glob(f"./{path}/*")
    print("Files will be delete.. \n", files)
    for f in files:
        os.remove(f)


def isBase64(sb):
    try:
        if isinstance(sb, str):
            sb_bytes = bytes(sb, "ascii")
        elif isinstance(sb, bytes):
            sb_bytes = sb
        else:
            raise ValueError("Argument must be string or bytes")
        return base64.b64encode(base64.b64decode(sb_bytes)) == sb_bytes
    except Exception:
        return False


@app.post("/upload")
def read_item(item: Item):

    clear_dir(inference_path)
    clear_dir(run_path)

    if item.categoryName == "8":
        if isBase64(item.imageSource) & isBase64(item.selectedPortion):
            print("--" * 50)
            print("\n cat 8")
            print("--" * 50)

            souce_file = base64.b64decode(item.imageSource)
            selected_templ_file = base64.b64decode(item.selectedPortion)

            file_name = f"source_img_{rd.randint(1, 500)}.jpg"
            print("-" * 50, "\n File uploaded with name: ", file_name, "-" * 50)

            templ_file_name = f"templ_source_img_{rd.randint(500, 999)}.jpg"
            print("-" * 50, "\n Templ File name: ", templ_file_name, "-" * 50)

            with open(os.path.join(os.getcwd(), inference_path, file_name), "wb") as f:
                f.write(souce_file)

            with open(
                os.path.join(os.getcwd(), inference_path, templ_file_name), "wb"
            ) as f:
                f.write(selected_templ_file)

            return image_match_count(file_name=file_name, template_name=templ_file_name)

        else:
            return "Image is not in base64-encoded format"
    else:

        if isBase64(item.imageSource):
            souce_file = base64.b64decode(item.imageSource)
            selected_templ_file = ""
            if item.selectedPortion != None:
                selected_templ_file = base64.b64decode(item.selectedPortion)

            file_name = f"source_img_{rd.randint(1, 500)}.jpg"
            print("-" * 50, "\n File uploaded with name: ", file_name, "-" * 50)

            templ_file_name = f"templ_source_img_{rd.randint(500, 999)}.jpg"
            print(
                "-" * 50,
                "\n Templ File uploaded with name: ",
                templ_file_name,
                "-" * 50,
            )

            with open(os.path.join(os.getcwd(), inference_path, file_name), "wb") as f:
                f.write(souce_file)

            if selected_templ_file != None:
                with open(
                    os.path.join(os.getcwd(), inference_path, templ_file_name), "wb"
                ) as f:
                    f.write(selected_templ_file)
            return generate_inferences(file_name, item.categoryName)
        else:
            return "Image is not in base64-encoded format"
