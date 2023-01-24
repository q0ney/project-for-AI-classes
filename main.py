import pygame   # importowanie potrzebnych bibliotek
import random
import os
import neat
pygame.font.init()

GEN = 0         # ustawienie parametrow wejsciowych i zaladowanie modelow
WIN_WIDTH = 800
WIN_HEIGHT = 800
win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
BG_IMG = pygame.image.load(os.path.join("images", "background.jpg"))
BARRIER_IMG = pygame.transform.scale(pygame.image.load(os.path.join("images", "police_car.png")), (100, 200))
CAR_IMG = pygame.transform.scale(pygame.image.load(os.path.join("images", "smart_car.png")), (100, 200))
STAT_FONT = pygame.font.SysFont("verdana", 50)


class Barrier:    # klasa przeszkody i jej metody
    IMG = BARRIER_IMG
    VEL = 30    # predkosc przeszkody

    def __init__(self):     # losowo generowane koordynaty przeszkod i ich wymiary
        self.x = random.randrange(0, WIN_WIDTH - 100)
        self.y = -self.IMG.get_height() - 100
        self.height = self.IMG.get_height()
        self.width = self.IMG.get_width()

    def draw(self, win):    # rysowanie przeszkod
        win.blit(self.IMG, (self.x, self.y))

    def move(self):      # poruszanie sie przeszkod z gory na dol
        self.y += self.VEL

    def collide(self, car):     # stworzenie masek modeli i punktu kolizyjnego dwoch masek
        car_mask = car.get_mask()
        barrier_mask = pygame.mask.from_surface(self.IMG)
        offset = (round(car.x) - self.x, car.Y_CAR - self.y)
        point = barrier_mask.overlap(car_mask, offset)
        if point:
            return True
        return False


class Car:  # klasa samochodu
    IMG = CAR_IMG
    Y_CAR = 570  # wysokosc na ktorej spawnia sie samochody
    VEL = 20
    MAX_ROTATION = 6
    ROT_VEL = 6
    ANIMATION_TIME = 5

    def __init__(self, x):      # parametry samochodu i miejsce z ktorego zaczyna
        self.x = x
        self.y = self.Y_CAR
        self.tilt = 0           # nachylenie pojazdu
        self.width = self.IMG.get_width()
        self.height = self.IMG.get_height()

    def move(self, direction):  # ruch pojazdu (rotacja i skrecanie)
        # jezeli wartosc jest TRUE pojazd skreca w prawo, jesli FALSE skreca w lewo
        if direction == "right" and self.x < WIN_WIDTH - CAR_IMG.get_width():
            self.x += self.VEL
            if self.tilt > -20:
                self.tilt -= self.ROT_VEL + 10
        elif direction == "left" and self.x > 0:
            self.x -= self.VEL
            if self.tilt < 20:
                self.tilt += self.ROT_VEL + 10

    def draw(self, win):        # rysowanie samochodu oraz jego poruszania sie na boki
        rotated_image = pygame.transform.rotate(self.IMG, self.tilt)  # obrót zdjęcia
        new_rect = rotated_image.get_rect(center=self.IMG.get_rect(topleft=(self.x, self.Y_CAR)).center)  # przesuniecie, ustawienie nowych wektorów kolizyjnych, "fizyczny" samochod w nowym miejscu
        win.blit(rotated_image, new_rect.topleft)

    def get_mask(self):     # pobranie maski naszego samochodu
        return pygame.mask.from_surface(self.IMG)


class Road:     # klasa drogi
    IMG = BG_IMG
    VEL = 30
    IMG_HEIGHT = IMG.get_height()

    def __init__(self, y):
        self.y = y

    def move(self):     # "ruch" drogi
        self.y += self.VEL
        if self.y > self.IMG_HEIGHT:
            self.y = -self.IMG_HEIGHT

    def draw(self, win):    # narysowanie drogi
        win.blit(self.IMG, (0, self.y))


def draw_window(win, roads, score, cars, barriers, pop, gen):     # klasa statystyk
    for road in roads:
        road.draw(win)
    for barrier in barriers:
        barrier.draw(win)
    for car in cars:
        car.draw(win)
    text1 = STAT_FONT.render("Score: " + str(score), 0, (255, 255, 255))
    text2 = STAT_FONT.render("Generation: " + str(gen - 1), 0, (255, 255, 255))
    text3 = STAT_FONT.render("Population: " + str(pop), 0, (255, 255, 255))
    win.blit(text1, (30, 10))
    win.blit(text2, (30, 10 + text1.get_height()))
    win.blit(text3, (30, 10 + text2.get_height() + text1.get_height()))
    pygame.display.update()


def main(genomes, config):  # glowna funkcja
    global GEN
    GEN += 1
    roads = []    # ustawienie potrzebnych tablic, do ktorych beda dodawne dane
    barriers = []
    nets = []
    ge = []
    cars = []
    for _, g in genomes:
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append(Car(WIN_WIDTH / 2 - CAR_IMG.get_width() / 2))
        ge.append(g)

    roads.append(Road(0))
    roads.append(Road(-800))
    score = 0

    clock = pygame.time.Clock()     # ustawienie czasu w jakim zdobywa sie pkt
    run = True
    while run:
        clock.tick(30)
        score += 1

        for event in pygame.event.get():        # skonczenie programu w momencie wyjscia
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()

        if len(cars) > 0:
            if len(barriers) > 0:
                for x, car in enumerate(cars):
                    ge[x].fitness += 0.1    # zwiekszenie fitnessu za kazda klatke (frame) ktora samochod przezyje
                    b = barriers[0]
                    #                          srodek samochodu          odleglosc y samochodu od przeszkody         odleglosc x samochpdu od przeszkody     odleglosc przeszkody od samochodu    odleglosc x sciany od przeszkody
                    output = nets[x].activate((abs(car.x + car.width / 2), abs(car.y - (b.y + b.height)),
                                               abs(car.x + car.width - (b.x + b.width)), abs(b.x), abs(b.x - car.x),
                                               abs(WIN_WIDTH - (b.x + b.width))))
                    best = output[0]
                    output[1] = output[1] + 0.001   # poprawienie odleglosci y samochodu od przeszkody
                    ind_best = 0
                    for index, o in enumerate(output):  # wybieranie najlepszej opcji ruchu samochodu
                        if o > best:
                            best = o
                            ind_best = index
                    if ind_best == 0:
                        car.move("left")
                    elif ind_best == 1:
                        car.move("right")
        else:
            run = False
            break

        if len(barriers) == 0:    # jezeli tablica barriers jest pusta dodajemy do niej informacje o przeszkodzie
            barriers.append(Barrier())
        for x, barrier in enumerate(barriers):
            barrier.move()
            if barrier.y > WIN_HEIGHT:    # jesli przeszkoda przekroczy wysokosc okienka, znika
                barriers.pop(x)
            for x, car in enumerate(cars):
                if barrier.y > car.y + car.height:    # jezeli samochod "ominie" przeszkode (jego y bedzie mniejszy) zwieksza sie fitness
                    ge[x].fitness += 5
                if car.x < 0 or car.x + car.height > WIN_WIDTH or barrier.collide(car):  # jezeli samochod zderzy sie z przeszkoda albo sciana, fitness sie zmniejsza i samochod znika oraz info o nim z tablic
                    ge[x].fitness -= 3
                    cars.pop(x)
                    nets.pop(x)
                    ge.pop(x)

        for road in roads:      # "ruch" drogi
            road.move()

        draw_window(win, roads, score, cars, barriers, len(cars), GEN)    # wyswietlenie okienka programu


def run(config_path):   # zaladowanie configu z neata i wypisanie go w konsoli przy kazdej generacji
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.run(main, 50)     # maksymalna ilosc generacji


if __name__ == "__main__":      # rozpznannie plikow w katalogu
    localdir = os.path.dirname(__file__)
    config_path = os.path.join(localdir, "config.txt")
    run(config_path)
