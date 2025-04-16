import os  
import time
import random
import math
import cv2
import keyboard
import mss
import numpy as np
import pygetwindow as gw
import win32api
import win32con
import warnings
from pywinauto import Application
from colorama import Fore, Style, init

# Инициализируем colorama
init(autoreset=True)

# Интервал проверки кнопки "Play" в секундах
CHECK_INTERVAL = 5

warnings.filterwarnings("ignore", category=UserWarning, module='pywinauto')

def list_windows_by_title(title):
    windows = gw.getAllWindows()
    filtered_windows = []
    for window in windows:
        if title.lower() in window.title.lower():
            filtered_windows.append((window.title, window._hWnd))
    return filtered_windows

class Logger:
    def __init__(self, prefix=None):
        self.prefix = prefix

    def log(self, data: str):
        if self.prefix:
            print(f"{self.prefix} {data}")
        else:
            print(data)

class AutoClicker:
    def __init__(self, hwnd, target_colors_hex, bomb_colors_hex, nearby_colors_hex, threshold, logger, target_percentage):
        self.hwnd = hwnd
        self.target_colors_hex = target_colors_hex
        self.bomb_colors_hex = bomb_colors_hex
        self.nearby_colors_hex = nearby_colors_hex
        self.threshold = threshold
        self.logger = logger
        self.target_percentage = target_percentage
        self.running = False
        self.clicked_points = []
        self.iteration_count = 0
        self.last_check_time = time.time()
        self.ignore_area_height = 50  # Высота области, где игнорируются клики

    @staticmethod
    def hex_to_hsv(hex_color):
        hex_color = hex_color.lstrip('#')
        h_len = len(hex_color)
        rgb = tuple(int(hex_color[i:i + h_len // 3], 16) for i in range(0, h_len, h_len // 3))
        rgb_normalized = np.array([[rgb]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_normalized, cv2.COLOR_RGB2HSV)
        return hsv[0][0]

    @staticmethod
    def click_at(x, y):
        try:
            win32api.SetCursorPos((x, y))
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, x, y, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, x, y, 0, 0)
        except Exception as e:
            print(f"Ошибка при установке позиции курсора: {e}")

    def toggle_script(self):
        self.running = not self.running
        r_text = "вкл" if self.running else f"{Fore.GREEN}выкл"
        self.logger.log(f'Статус изменен: {r_text}')

    def check_and_click_play_button(self, sct, monitor):
        current_time = time.time()
        if current_time - self.last_check_time >= CHECK_INTERVAL:
            self.last_check_time = current_time
            templates = [
                cv2.imread(os.path.join("template_png", "template_play_button.png"), cv2.IMREAD_GRAYSCALE),
                cv2.imread(os.path.join("template_png", "template_play_button1.png"), cv2.IMREAD_GRAYSCALE),
                cv2.imread(os.path.join("template_png", "close_button.png"), cv2.IMREAD_GRAYSCALE),
                cv2.imread(os.path.join("template_png", "captcha.png"), cv2.IMREAD_GRAYSCALE)
            ]

            img = np.array(sct.grab(monitor))
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

            for template in templates:
                if template is None:
                    self.logger.log(Fore.RED + "Не удалось загрузить файл шаблона.")
                    continue

                template_height, template_width = template.shape
                res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= self.threshold)
                matched_points = list(zip(*loc[::-1]))

                if matched_points:
                    pt_x, pt_y = matched_points[0]
                    cX = pt_x + template_width // 2 + monitor["left"]
                    cY = pt_y + template_height // 2 + monitor["top"]

                    self.click_at(cX, cY)
                    self.logger.log(f'Нажал на кнопку: {cX} {cY}')
                    self.clicked_points.append((cX, cY))
                    break

    def is_near_color(self, hsv, point, hsv_nearby):
        color_at_point = hsv[point[1], point[0]]
        for nearby_color in hsv_nearby:
            if all(abs(int(color_at_point[i]) - int(nearby_color[i])) < 30 for i in range(3)):
                return True
        return False

    def click_color_areas(self):
        app = Application().connect(handle=self.hwnd)
        window = app.window(handle=self.hwnd)
        window.set_focus()

        target_hsvs = [self.hex_to_hsv(color) for color in self.target_colors_hex]
        bomb_hsvs = [self.hex_to_hsv(color) for color in self.bomb_colors_hex]
        nearby_hsvs = [self.hex_to_hsv(color) for color in self.nearby_colors_hex]

        with mss.mss() as sct:
            keyboard.add_hotkey('F6', self.toggle_script)

            while True:
                if self.running:
                    # Проверка активного окна
                    active_window = gw.getActiveWindow()
                    if active_window._hWnd != self.hwnd:
                        self.logger.log(Fore.RED + "Фокус не на нужном окне.")
                        time.sleep(0.5)
                        continue

                    rect = window.rectangle()
                    monitor = {
                        "top": rect.top,
                        "left": rect.left,
                        "width": rect.width(),
                        "height": rect.height()
                    }

                    self.check_and_click_play_button(sct, monitor)

                    img = np.array(sct.grab(monitor))
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

                    # Обработка тыкв
                    self.process_contours(hsv, target_hsvs, nearby_hsvs, monitor, "тыквы")

                    # Обработка бомб
                    self.process_bomb_contours(hsv, monitor)

                    time.sleep(0.1)
                    self.iteration_count += 1

                    if self.iteration_count >= 5:
                        self.clicked_points.clear()
                        self.iteration_count = 0

    def process_contours(self, hsv, hsv_targets, hsv_nearby, monitor, target_name):
        for target_hsv in hsv_targets:
            lower_bound = np.array([max(0, target_hsv[0] - 10), 100, 100])
            upper_bound = np.array([min(179, target_hsv[0] + 10), 255, 255])
            mask = cv2.inRange(hsv, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                num_contours = len(contours)
                num_to_click = max(1, int(num_contours * self.target_percentage))
                contours_to_click = random.sample(contours, min(num_to_click, num_contours))

                for contour in reversed(contours_to_click):
                    if cv2.contourArea(contour) < 50:
                        continue

                    M = cv2.moments(contour)
                    if M["m00"] == 0:
                        continue
                    cX = int(M["m10"] / M["m00"]) + monitor["left"]
                    cY = int(M["m01"] / M["m00"]) + monitor["top"]

                    # Пропуск клика, если в верхней зоне
                    if cY < monitor["top"] + self.ignore_area_height:
                        continue

                    if not self.is_near_color(hsv, (cX - monitor["left"], cY - monitor["top"]), hsv_nearby):
                        continue

                    if any(math.sqrt((cX - px) ** 2 + (cY - py) ** 2) < 40 for px, py in self.clicked_points):
                        continue

                    self.click_at(cX, cY)
                    self.logger.log(f'Нажал на {target_name}: {cX} {cY}')
                    self.clicked_points.append((cX, cY))

    def process_bomb_contours(self, hsv, monitor):
        lower_bound = np.array([0, 0, 150])  # Строго серые оттенки, отсекая цвета интерфейса
        upper_bound = np.array([180, 50, 220])
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) < 30:
                continue

            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"]) + monitor["left"]
            cY = int(M["m01"] / M["m00"]) + monitor["top"]

            if cY < monitor["top"] + self.ignore_area_height:
                continue

            if any(math.sqrt((cX - px) ** 2 + (cY - py) ** 2) < 40 for px, py in self.clicked_points):
                continue

            self.click_at(cX, cY)
            self.logger.log(f'Нажал на бомбу: {cX} {cY}')
            self.clicked_points.append((cX, cY))

def print_custom_banner():
    banner = """
 __      __                 __   _________.__    .___      
/  \    /  \ ____   _______/  |_/   _____/|__| __| _/____  
\   \/\/   // __ \ /  ___/\   __\_____  \ |  |/ __ |/ __ \ 
 \        /\  ___/ \___ \  |  | /        \|  / /_/ \  ___/ 
  \__/\  /  \___  >____  > |__|/_______  /|__\____ |\___  >
       \/       \/     \/              \/         \/    \/ 
"""
    print(banner)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)

    print_custom_banner()  # Вывод баннера

    title = "TelegramDesktop"  # Фильтр для окна Telegram
    windows = list_windows_by_title(title)

    if not windows:
        print(Fore.RED + "Нет окон, содержащих указанные ключевые слова.")  # Красный текст
        exit()

    print("Доступные окна для выбора:")  
    for i, (title, hwnd) in enumerate(windows):
        print(f"{i + 1}: {title}")

    choice = int(input("Введите номер окна, в котором открыт бот Blum: ")) - 1
    if choice < 0 or choice >= len(windows):
        print(Fore.RED + "Неверный выбор.")  # Красный текст
        exit()

    hwnd = windows[choice][1]
    print(Fore.RED + "не будь ленивой сракой ёпта ,просто введи значение и тыкни F6 для запуска дурмашины")  # Красный текст

    while True:
        try:
            target_percentage = input(Fore.YELLOW + "Введи значение от 0 до 1 для рандомизации прокликивания тыкв, где 1 означает сбор всех тыкв. (Рекомендуемое значение: 0.09) : ") 
            target_percentage = target_percentage.replace(',', '.')
            target_percentage = float(target_percentage)
            if 0 <= target_percentage <= 1:
                break
            else:
                print(Fore.RED + "Пожалуйста, введите значение от 0 до 1.")  # Красный текст
        except ValueError:
            print(Fore.RED + "Некорректный ввод. Пожалуйста, введите число.")  # Красный текст

    logger = Logger("[DEBUG]")

    # Настройки цветов
    target_colors_hex = ["#ff8c00", "#ff4500"]  # Основные цвета для оранжевых тыкв
    bomb_colors_hex = ["#c0c0c0", "#ffffff"]    # Цвета для бомбочек
    nearby_colors_hex = ["#ff7f50", "#ffa07a"]  # Близкие оттенки

    # Инициализация и запуск AutoClicker с учетом бомбочек
    auto_clicker = AutoClicker(hwnd=hwnd,
                               target_colors_hex=target_colors_hex,
                               bomb_colors_hex=bomb_colors_hex,
                               nearby_colors_hex=nearby_colors_hex,
                               threshold=0.8,  # Порог совпадения для поиска объектов
                               logger=logger,
                               target_percentage=target_percentage)

    auto_clicker.click_color_areas()  # Запуск процесса отслеживания и нажатия
