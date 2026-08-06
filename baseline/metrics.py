def mean_absdiff(np_module, img1, img2):
    return float(np_module.mean(np_module.abs(img1.astype(np_module.float32) - img2.astype(np_module.float32))))

def psnr(np_module, img1, img2):
    mse = np_module.mean((img1.astype(np_module.float32) - img2.astype(np_module.float32)) ** 2)
    if mse == 0:
        return 100.0
    return float(20.0 * np_module.log10(255.0 / np_module.sqrt(mse)))

def ssim_gray(cv2_module, np_module, img1, img2):
    from skimage.metrics import structural_similarity as ssim
    if len(img1.shape) == 3 and img1.shape[2] == 3:
        g1 = cv2_module.cvtColor(img1, cv2_module.COLOR_RGB2GRAY)
        g2 = cv2_module.cvtColor(img2, cv2_module.COLOR_RGB2GRAY)
    else:
        g1 = img1
        g2 = img2
    return float(ssim(g1, g2, data_range=255.0))
