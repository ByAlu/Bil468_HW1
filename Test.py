import TemplateMatcher

if __name__ == "__main__":
   
    matcher = TemplateMatcher.TemplateMatcher()
    # en iyi değerlere ayarla
    matcher.threshold = 0.7
    matcher._method_threshold = 0.9
    matcher.test()