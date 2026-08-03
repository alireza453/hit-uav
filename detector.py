import onnxruntime as ort
import cv2
import numpy as np
import os

class YOLOv8ONNX:
    def __init__(
        self,
        model_path,
        class_names,
        conf_thres=0.25,
        iou_thres=0.45,
    ):

        self.class_names = class_names
        self.conf = conf_thres
        self.iou = iou_thres

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        _, _, self.input_h, self.input_w = \
            self.session.get_inputs()[0].shape

    ####################################################
    # Preprocess
    ####################################################

    def preprocess(self, image):

        self.orig_h, self.orig_w = image.shape[:2]

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resized = cv2.resize(
            rgb,
            (self.input_w, self.input_h)
        )

        resized = resized.astype(np.float32) / 255.0

        resized = resized.transpose(2, 0, 1)

        return np.expand_dims(resized, axis=0)

    ####################################################
    # Inference
    ####################################################

    def infer(self, image):

        input_tensor = self.preprocess(image)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: input_tensor}
        )

        return outputs[0]

    ####################################################
    # Postprocess
    ####################################################

    def postprocess(self, image, output):

        predictions = output.squeeze().T

        x_factor = self.orig_w / self.input_w
        y_factor = self.orig_h / self.input_h

        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:

            cx, cy, w, h = pred[:4]

            cls_scores = pred[4:]

            class_id = np.argmax(cls_scores)

            score = cls_scores[class_id]

            if score < self.conf:
                continue

            left = int((cx - w / 2) * x_factor)
            top = int((cy - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * y_factor)

            boxes.append([left, top, width, height])
            scores.append(float(score))
            class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.conf,
            self.iou
        )

        if len(indices):

            for idx in indices.flatten():

                x, y, w, h = boxes[idx]

                cv2.rectangle(
                    image,
                    (x, y),
                    (x + w, y + h),
                    (255, 255, 0),
                    1
                )

                label = "{} {:.2f}".format(
                    self.class_names[class_ids[idx]],
                    scores[idx]
                )

                (tw, th), _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2
                )

                cv2.rectangle(
                    image,
                    (x, y - 25),
                    (x + tw + 8, y),
                    (255, 255, 0),
                    -1
                )

                cv2.putText(
                    image,
                    label,
                    (x + 4, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    1
                )

        return image

    ####################################################
    # Detect Image
    ####################################################

    def detect_image(self, image_path, save_path):

        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(image_path)

        output = self.infer(image)

        result = self.postprocess(image.copy(), output)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        cv2.imwrite(save_path, result)

        return save_path

    ####################################################
    # Detect Video
    ####################################################
    def detect_video(
        self,
        input_path,
        output_path,
        progress_callback=None
    ):

        cap = cv2.VideoCapture(input_path)

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        frame = 0

        while True:

            ret, img = cap.read()

            if not ret:
                break

            output = self.infer(img)

            img = self.postprocess(img, output)

            writer.write(img)

            frame += 1

            if progress_callback:
                progress_callback(frame, total)

        cap.release()
        writer.release()