import os
import numpy as np
import pandas as pd
import cv2
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error
)


class TemplateMatcher:
    def __init__(self, templates_dir="hw1_dataset/training_images/", validationSet="hw1_dataset/labels_validation.csv", testSet="hw1_dataset/labels_test.csv"):
        self._method = cv2.TM_CCOEFF_NORMED
        self._method_threshold = 0.7
        self._kernelSize = 3
        
        self.templates = self.load_templates(templates_dir)
        self.validationSet = validationSet
        self.testSet = testSet
        self.threshold = 0.5
        
       

    def load_templates(self, templates_dir):
        templates = {}
        for template_name in os.listdir(templates_dir):
            template_path = os.path.join(templates_dir, template_name)
            template_image = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            templates[template_name] = []
            template_image = cv2.GaussianBlur(template_image, (self._kernelSize, self._kernelSize), 0)
            sizes = [256, 128, 64, 32, 16, 8]
            for size in sizes:
                resized_template = cv2.resize(template_image, (size, size))
                templates[template_name].append(resized_template)
                    

        return templates
    
    # tm yi uygullar ve metoda göre değer ve lokasyonu döner
    def apply_template_matching(self_,image, template):
        # imageye önişlem


        result = cv2.matchTemplate(image, template, self_._method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if self_._method == cv2.TM_CCOEFF_NORMED or self_._method == cv2.TM_CCORR_NORMED:
            return max_val, max_loc
        else:
            return min_val, min_loc
    
    # tm metoduna göre sonuç karşılaştırması yapar
    def compare_results(self_, prior, new):
        if self_._method == cv2.TM_CCOEFF_NORMED or self_._method == cv2.TM_CCORR_NORMED:
            return new > prior
        else:
            return new < prior

    # x_min, x_max, y_min, y_max
    def calculate_area_intersection(self, trueArea, foundArea):
        x_min = max(trueArea[0], foundArea[0])
        y_min = max(trueArea[1], foundArea[1])
        x_max = min(trueArea[2], foundArea[2])
        y_max = min(trueArea[3], foundArea[3])

        interArea = max(0, x_max - x_min) * max(0, y_max - y_min)
        trueAreaSize = (trueArea[2] - trueArea[0]) * (trueArea[3] - trueArea[1])
        return interArea/trueAreaSize if trueAreaSize > 0 else 0
    
    # gaussian blur için kernel size set etmek için bunu kullan. 
    def set_kernel_size(self_, size):
        self_._kernelSize = size
        self_.load_templates()

    def load_images_from_csv(self, csv):
        data = pd.read_csv(csv)
        image_paths = data["frame"].values
        return_iamges = []
        for path in image_paths:
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            image = cv2.GaussianBlur(image, (self._kernelSize, self._kernelSize), 0)
            image = cv2.resize(image,(256,256))
            return_iamges.append(image)
        
        return return_iamges 

    def train(self, thresholds, method_thresholds):
        return_scores = []
        best_f1 = -1
        validation_data = pd.read_csv(self.validationSet)
        rows = validation_data.shape[0]
        y_true = validation_data["contains_car"].values
        best_threshold = 0
        best_method_threshold = 0
        images = self.load_images_from_csv(self.validationSet)
        for method_threshold in method_thresholds:
            for threshold in thresholds:
                y_pred = []
                for i in range(rows):
                    image = images[i]
                    best_intersection = [method_threshold, (0,0), (0,0)]
                    for(template_name, template_list) in self.templates.items():
                        for template in template_list:
                            val, loc = self.apply_template_matching(image, template)
                                                    
                            if self.compare_results(best_intersection[0], val):
                                best_intersection = [val,loc,template.shape]

                    if ~self.compare_results(method_threshold, best_intersection[0]):
                        y_pred.append(False)
                        continue
                    else:
                        if(y_true[i]):
                            min_loc = best_intersection[1]
                            template = best_intersection[2]
                            intersect_area = self.calculate_area_intersection(
                                        (validation_data.loc[i]["xmin"], validation_data.loc[i]["ymin"],
                                        validation_data.loc[i]["xmax"], validation_data.loc[i]["ymax"]),
                                        (min_loc[0], min_loc[1], min_loc[0] + template[1], min_loc[1] + template[0])
                                    )
                            if intersect_area >= threshold:
                                y_pred.append(True)
                            else:
                                y_pred.append(False)
                        else:
                            y_pred.append(False)
                
                # confusion matrix hesapla
                cm = confusion_matrix(y_true, y_pred)
                acs = accuracy_score(y_true, y_pred)
                ps = precision_score(y_true, y_pred)
                rs = recall_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)

                return_scores.append({
                    "method_threshold":method_threshold,
                    "threshold": threshold,                  
                    "confusion_matrix":cm,
                    "accuracy_score": acs,
                    "precision_score": ps,
                    "recall_score": rs,
                    "f1_score":f1,
                    "mean_squared_error": mse
                } 
                )

                if(f1>best_f1):
                    best_threshold = threshold
                    best_method_threshold = method_threshold

        
        self.threshold = best_threshold
        self._method_threshold = best_method_threshold
        return return_scores
    

    def evaluate(self):
        validation_data = pd.read_csv(self.testSet)
        rows = validation_data.shape[0]
        y_true = validation_data["contains_car"].values
        y_pred = []
        images = self.load_images_from_csv(self.testSet)
        for i in range(rows):
            image = images[i]
            best_intersection = [self._method_threshold, (0,0), (0,0)]
            for(template_name, template_list) in self.templates.items():
                for template in template_list:
                    val, loc = self.apply_template_matching(image, template)                                               
                    if self.compare_results(best_intersection[0], val):
                        best_intersection = [val,loc,template.shape]

            if ~self.compare_results(self._method_threshold, best_intersection[0]):
                y_pred.append(False)
                continue
            else:
                if(y_true[i]):
                    min_loc = best_intersection[1]
                    template = best_intersection[2]
                    intersect_area = self.calculate_area_intersection(
                                (validation_data.loc[i]["xmin"], validation_data.loc[i]["ymin"],
                                validation_data.loc[i]["xmax"], validation_data.loc[i]["ymax"]),
                                (min_loc[0], min_loc[1], min_loc[0] + template[1], min_loc[1] + template[0])
                            )
                    if intersect_area >= self.threshold:
                        y_pred.append(True)
                    else:
                        y_pred.append(False)
                else:
                    y_pred.append(False)

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)

        confusion = confusion_matrix(y_true, y_pred)
        return confusion, accuracy, precision, recall, f1, mse
    

    # Kullanılmadan önce threshold değeri belirlenmelidir. Tüm test verisi için çalışır 
    def test(self):
        test_data = pd.read_csv(self.testSet)
        rows = test_data.shape[0]
        y_true = test_data["contains_car"].values
        for i in range(rows):
            imageName = test_data.iloc[i]["frame"]
            print(imageName, " ", y_true[i])
            image = cv2.imread(imageName)
            image_sub = cv2.resize(image,(256,256))
            image_gray = cv2.imread(imageName, cv2.IMREAD_GRAYSCALE)
            image_gray = cv2.GaussianBlur(image_gray, (self._kernelSize, self._kernelSize), 0)
            image_gray = cv2.resize(image_gray,(256,256))
            best_intersection = [self._method_threshold, (0,0), (0,0)]
            for(template_name, template_list) in self.templates.items():
                for template in template_list:
                    val, loc = self.apply_template_matching(image_gray, template)
                    #bulundu mu
                    if self.compare_results(best_intersection[0], val):
                        best_intersection = [val, loc, template.shape]
                        
            # Doğru alan mavi, bulunan alan yeşil
            if self.compare_results(self._method_threshold, best_intersection[0]):
                print(best_intersection)
                min_loc = best_intersection[1]
                template = best_intersection[2]
                cv2.rectangle(image_sub, (min_loc[0], min_loc[1]), (min_loc[0] + template[0], min_loc[1] + template[1]), (0, 255, 0), 2)

                cv2.imshow("Detected", image_sub)
                key = cv2.waitKey(0)
                if(key == 27):  # ESC tuşu
                    cv2.destroyAllWindows()
                    break         








