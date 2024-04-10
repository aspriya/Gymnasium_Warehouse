import sys, pygame
pygame.init() # 

size = width, height = 1024, 768
speed = [1, 1]
black = 255, 255, 255

screen = pygame.display.set_mode(size)

pallet_jack = pygame.image.load("warehoue_worker.png")
# zoom out the image
pallet_jack = pygame.transform.scale(pallet_jack, (30, 40))
pallet_jack_rect = pallet_jack.get_rect()

while 1:

    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    pallet_jack_rect = pallet_jack_rect.move(speed)
    if pallet_jack_rect.left < 0 or pallet_jack_rect.right > width:
        speed[0] = -speed[0]
    if pallet_jack_rect.top < 0 or pallet_jack_rect.bottom > height:
        speed[1] = -speed[1]

    screen.fill(black) # fill the screen with black color
    screen.blit(pallet_jack, pallet_jack_rect) # set the pallet jack and its rect on the screen
    pygame.display.flip() # update the screen

    pygame.time.delay(100) # delay the screen update by 10 ms