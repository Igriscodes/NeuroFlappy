#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <deque>
#include <raylib.h>

constexpr int VIRTUAL_WIDTH = 800;
constexpr int VIRTUAL_HEIGHT = 600;
constexpr int POP_SIZE = 250;

constexpr float BIRD_RADIUS = 12.0f;
constexpr float GRAVITY = 0.6f;
constexpr float FLAP_STRENGTH = -8.0f;
constexpr float PIPE_SPEED = 4.0f;
constexpr float PIPE_WIDTH = 60.0f;
constexpr float PIPE_GAP = 140.0f;
constexpr int SPAWN_RATE = 90;

constexpr int NUM_INPUTS = 5;
constexpr int NUM_HIDDEN = 8;
constexpr int NUM_OUTPUTS = 1;
constexpr int NUM_WEIGHTS = (NUM_INPUTS * NUM_HIDDEN) + NUM_HIDDEN + (NUM_HIDDEN * NUM_OUTPUTS) + NUM_OUTPUTS;

std::mt19937 rng(std::random_device{}());
double randf(double min, double max) { return std::uniform_real_distribution<double>(min, max)(rng); }
double randf() { return randf(0.0, 1.0); }

struct Genome {
    std::vector<double> weights;
    double fitness = 0.0;

    Genome() {
        weights.resize(NUM_WEIGHTS);
        for (double& w : weights) w = randf(-1.0, 1.0);
    }

    void mutate() {
        for (double& w : weights) {
            if (randf() < 0.10) w += randf(-0.5, 0.5);
            if (randf() < 0.02) w = randf(-2.0, 2.0);
        }
    }

    static Genome crossover(const Genome& p1, const Genome& p2) {
        Genome child;
        for (size_t i = 0; i < NUM_WEIGHTS; i++) {
            child.weights[i] = (randf() < 0.5) ? p1.weights[i] : p2.weights[i];
        }
        return child;
    }
};

struct Bird {
    float y = VIRTUAL_HEIGHT / 2.0f;
    float vy = 0.0f;
    bool alive = true;
    double fitness = 0.0;
    
    void flap() { vy = FLAP_STRENGTH; }
};

struct Pipe {
    float x;
    float gapTop, gapBottom;
};

struct Agent {
    Genome genome;
    Bird bird;
    bool flapPrev = false;
    
    std::vector<double> hiddenActivations;
    double outputActivation = 0.0;

    Agent() { hiddenActivations.resize(NUM_HIDDEN, 0.0); }

    double evaluate(const double* inputs) {
        int wIdx = 0;
        
        for (int i = 0; i < NUM_HIDDEN; i++) {
            double sum = 0.0;
            for (int j = 0; j < NUM_INPUTS; j++) {
                sum += inputs[j] * genome.weights[wIdx++];
            }
            sum += genome.weights[wIdx++];
            hiddenActivations[i] = std::tanh(sum);
        }
        
        double sum = 0.0;
        for (int i = 0; i < NUM_HIDDEN; i++) {
            sum += hiddenActivations[i] * genome.weights[wIdx++];
        }
        sum += genome.weights[wIdx++];
        
        outputActivation = 1.0 / (1.0 + std::exp(-sum));
        return outputActivation;
    }
};

std::vector<Agent> agents;
std::deque<Pipe> pipes;

int generation = 1;
int frames = 0;
double allTimeBestFitness = 0;

const int ALLOWED_SPEEDS[] = {1, 5, 10, 20, 50, 100};
int currentSpeedIdx = 0;

void resetGame() {
    frames = 0;
    pipes.clear();
    for (auto& a : agents) {
        a.bird = Bird();
        a.flapPrev = false;
    }
}

void nextGeneration() {
    std::sort(agents.begin(), agents.end(), [](const Agent& a, const Agent& b) {
        return a.genome.fitness > b.genome.fitness;
    });

    std::vector<Genome> nextGenomes;
    
    int elites = POP_SIZE * 0.05;
    for (int i = 0; i < elites; i++) {
        nextGenomes.push_back(agents[i].genome);
    }

    int topHalf = POP_SIZE / 2;
    while (nextGenomes.size() < POP_SIZE) {
        const Genome& p1 = agents[rand() % topHalf].genome;
        const Genome& p2 = agents[rand() % topHalf].genome;
        
        Genome child = Genome::crossover(p1, p2);
        child.mutate();
        nextGenomes.push_back(child);
    }

    for (int i = 0; i < POP_SIZE; i++) {
        agents[i].genome = nextGenomes[i];
    }
    
    generation++;
}

void updateGameLoop(int& aliveCount) {
    frames++;
    if (frames % SPAWN_RATE == 0) {
        float gapTop = randf(50.0f, VIRTUAL_HEIGHT - 50.0f - PIPE_GAP);
        pipes.push_back({(float)VIRTUAL_WIDTH, gapTop, gapTop + PIPE_GAP});
    }
    
    for (auto& p : pipes) p.x -= PIPE_SPEED;
    if (!pipes.empty() && pipes.front().x < -PIPE_WIDTH) pipes.pop_front();

    Pipe nextPipe = {(float)VIRTUAL_WIDTH + 100.0f, 200.0f, 350.0f}; 
    for (auto& p : pipes) {
        if (p.x + PIPE_WIDTH > 50) { nextPipe = p; break; }
    }

    aliveCount = 0;
    double inputs[NUM_INPUTS];

    for (auto& a : agents) {
        if (!a.bird.alive) continue;
        aliveCount++;
        
        inputs[0] = a.bird.y / VIRTUAL_HEIGHT;
        inputs[1] = (a.bird.vy + 12.0) / 24.0; 
        inputs[2] = (nextPipe.x - 50.0) / VIRTUAL_WIDTH;
        inputs[3] = nextPipe.gapTop / VIRTUAL_HEIGHT;
        inputs[4] = nextPipe.gapBottom / VIRTUAL_HEIGHT;
        
        double out = a.evaluate(inputs);
        if (out > 0.5 && !a.flapPrev) a.bird.flap();
        a.flapPrev = out > 0.5;

        a.bird.vy += GRAVITY;
        if (a.bird.vy > 12.0f) a.bird.vy = 12.0f;
        a.bird.y += a.bird.vy;
        
        a.bird.fitness += 1.0; 
        a.genome.fitness = a.bird.fitness;
        
        if (a.genome.fitness > allTimeBestFitness) allTimeBestFitness = a.genome.fitness;

        if (a.bird.y < 0 || a.bird.y > VIRTUAL_HEIGHT) a.bird.alive = false;
        if (a.bird.alive) {
            for (auto& p : pipes) {
                if (50 + BIRD_RADIUS > p.x && 50 - BIRD_RADIUS < p.x + PIPE_WIDTH) {
                    if (a.bird.y - BIRD_RADIUS < p.gapTop || a.bird.y + BIRD_RADIUS > p.gapBottom) {
                        a.bird.alive = false; break;
                    }
                }
            }
        }
    }
    
    if (aliveCount == 0) {
        nextGeneration();
        resetGame();
    }
}

int main() {
    SetConfigFlags(FLAG_VSYNC_HINT | FLAG_WINDOW_RESIZABLE);
    SetTraceLogLevel(LOG_WARNING);
    InitWindow(VIRTUAL_WIDTH, VIRTUAL_HEIGHT, "Neural Network AI: Flappy Bird");
    SetWindowMinSize(640, 480);
    SetTargetFPS(60);

    RenderTexture2D target = LoadRenderTexture(VIRTUAL_WIDTH, VIRTUAL_HEIGHT);
    SetTextureFilter(target.texture, TEXTURE_FILTER_BILINEAR);

    Texture2D bgTex = LoadTexture("assets/background.png");
    Texture2D birdTex = LoadTexture("assets/bird.png");
    Texture2D pipeTex = LoadTexture("assets/pipe.png");
    float bgScroll = 0.0f;

    agents.resize(POP_SIZE);

    int aliveCount = POP_SIZE;

    while (!WindowShouldClose()) {
        if (IsKeyPressed(KEY_F11) || (IsKeyDown(KEY_LEFT_ALT) && IsKeyPressed(KEY_ENTER))) {
            ToggleFullscreen();
        }

        if (IsKeyPressed(KEY_RIGHT)) currentSpeedIdx = std::min(5, currentSpeedIdx + 1);
        if (IsKeyPressed(KEY_LEFT))  currentSpeedIdx = std::max(0, currentSpeedIdx - 1);
        int speedMultiplier = ALLOWED_SPEEDS[currentSpeedIdx];

        for (int i = 0; i < speedMultiplier; i++) {
            updateGameLoop(aliveCount);
            if (aliveCount > 0) bgScroll -= (PIPE_SPEED * 0.25f); 
        }

        if (bgTex.width > 0) {
            float bgScale = (float)VIRTUAL_HEIGHT / bgTex.height;
            float bgWidth = bgTex.width * bgScale;
            if (bgScroll <= -bgWidth) bgScroll += bgWidth;
        }

        BeginTextureMode(target);
        ClearBackground(Color{135, 206, 235, 255}); 

        if (bgTex.id != 0) {
            float bgScale = (float)VIRTUAL_HEIGHT / bgTex.height;
            float bgWidth = bgTex.width * bgScale;
            DrawTextureEx(bgTex, {bgScroll, 0}, 0.0f, bgScale, WHITE);
            DrawTextureEx(bgTex, {bgScroll + bgWidth, 0}, 0.0f, bgScale, WHITE);
            DrawTextureEx(bgTex, {bgScroll + bgWidth * 2.0f, 0}, 0.0f, bgScale, WHITE);
        }

        for (auto& p : pipes) {
            if (pipeTex.id != 0) {
                Rectangle bottomDest = {p.x, p.gapBottom, PIPE_WIDTH, VIRTUAL_HEIGHT - p.gapBottom};
                Rectangle bottomSrc = {0, 0, (float)pipeTex.width, (float)pipeTex.height};
                DrawTexturePro(pipeTex, bottomSrc, bottomDest, {0,0}, 0.0f, WHITE);

                Rectangle topDest = {p.x, 0, PIPE_WIDTH, p.gapTop};
                Rectangle topSrc = {0, 0, (float)pipeTex.width, -(float)pipeTex.height};
                DrawTexturePro(pipeTex, topSrc, topDest, {0,0}, 0.0f, WHITE);
            } else {
                DrawRectangle(p.x, 0, PIPE_WIDTH, p.gapTop, Color{50, 205, 50, 255});
                DrawRectangle(p.x, p.gapBottom, PIPE_WIDTH, VIRTUAL_HEIGHT - p.gapBottom, Color{50, 205, 50, 255});
            }
        }

        Agent* bestAgent = &agents[0];
        double bestFit = -1;

        for (auto& a : agents) {
            if (!a.bird.alive) continue;
            if (a.genome.fitness > bestFit) { bestFit = a.genome.fitness; bestAgent = &a; }
            
            if (birdTex.id != 0) {
                float rotation = a.bird.vy * 4.0f;
                if (rotation < -45.0f) rotation = -45.0f;
                if (rotation > 45.0f) rotation = 45.0f;
                
                Rectangle src = {0, 0, (float)birdTex.width, (float)birdTex.height};
                Rectangle dest = {50.0f, a.bird.y, (float)birdTex.width * 1.5f, (float)birdTex.height * 1.5f};
                Vector2 origin = {dest.width / 2.0f, dest.height / 2.0f};
                
                DrawTexturePro(birdTex, src, dest, origin, rotation, Color{255, 255, 255, 100}); 
            } else {
                DrawCircle(50, a.bird.y, BIRD_RADIUS, Color{255, 255, 0, 100});
            }
        }
        
        if (bestAgent->bird.alive) {
            if (birdTex.id != 0) {
                float rotation = bestAgent->bird.vy * 4.0f;
                if (rotation < -45.0f) rotation = -45.0f;
                if (rotation > 45.0f) rotation = 45.0f;
                
                Rectangle src = {0, 0, (float)birdTex.width, (float)birdTex.height};
                Rectangle dest = {50.0f, bestAgent->bird.y, (float)birdTex.width * 1.5f, (float)birdTex.height * 1.5f};
                Vector2 origin = {dest.width / 2.0f, dest.height / 2.0f};
                
                DrawCircle(50, bestAgent->bird.y, BIRD_RADIUS * 2.0f, Color{255, 215, 0, 150}); 
                DrawTexturePro(birdTex, src, dest, origin, rotation, WHITE); 
            } else {
                DrawCircleLines(50, bestAgent->bird.y, BIRD_RADIUS + 2, RED);
                DrawCircle(50, bestAgent->bird.y, BIRD_RADIUS, Color{255, 165, 0, 255});
            }
        }

        DrawRectangleRounded(Rectangle{(float)VIRTUAL_WIDTH - 280, 10, 270, 270}, 0.1f, 10, Color{0, 0, 0, 180});
        DrawText("Deep Neural Network", VIRTUAL_WIDTH - 240, 20, 20, RAYWHITE);
        
        float startX = VIRTUAL_WIDTH - 240;
        float layerSpacing = 100.0f;
        
        Vector2 inNodes[NUM_INPUTS];
        for (int i=0; i<NUM_INPUTS; i++) inNodes[i] = {startX, 60.0f + i * 40.0f};
        
        Vector2 hNodes[NUM_HIDDEN];
        for (int i=0; i<NUM_HIDDEN; i++) hNodes[i] = {startX + layerSpacing, 50.0f + i * 27.0f};
        
        Vector2 outNode = {startX + layerSpacing * 2, 140.0f};

        int wIdx = 0;
        for (int i = 0; i < NUM_HIDDEN; i++) {
            for (int j = 0; j < NUM_INPUTS; j++) {
                double w = bestAgent->genome.weights[wIdx++];
                Color c = (w > 0) ? GREEN : RED;
                c.a = (unsigned char)(std::min(1.0, std::abs(w)) * 100);
                DrawLineEx(inNodes[j], hNodes[i], std::max(1.0, std::abs(w)*2.0), c);
            }
            wIdx++;
        }
        
        for (int i = 0; i < NUM_HIDDEN; i++) {
            double w = bestAgent->genome.weights[wIdx++];
            Color c = (w > 0) ? GREEN : RED;
            double act = std::max(0.0, bestAgent->hiddenActivations[i]);
            c.a = (unsigned char)(std::max(0.1, act) * 255); 
            DrawLineEx(hNodes[i], outNode, std::max(1.0, std::abs(w)*3.0), c);
        }

        for (int i=0; i<NUM_INPUTS; i++) DrawCircleV(inNodes[i], 6, SKYBLUE);
        for (int i=0; i<NUM_HIDDEN; i++) DrawCircleV(hNodes[i], 6, PURPLE);
        DrawCircleV(outNode, 8, PINK);
        
        for (int i=0; i<NUM_HIDDEN; i++) {
            float act = (bestAgent->hiddenActivations[i] + 1.0f) / 2.0f;
            DrawCircleV(hNodes[i], 4, Color{(unsigned char)(255*act), (unsigned char)(255*act), (unsigned char)(255*act), 255});
        }
        float outAct = bestAgent->outputActivation;
        DrawCircleV(outNode, 6, Color{(unsigned char)(255*outAct), (unsigned char)(255*outAct), (unsigned char)(255*outAct), 255});

        DrawRectangleRounded(Rectangle{10, 10, 260, 135}, 0.2f, 10, Fade(BLACK, 0.6f));
        DrawText(TextFormat("Generation: %d", generation), 20, 20, 20, WHITE);
        DrawText(TextFormat("Alive: %d / %d", aliveCount, POP_SIZE), 20, 45, 20, WHITE);
        
        int currentScore = bestFit > 0 ? (int)(bestFit / 10) : 0;
        DrawText(TextFormat("Score: %d", currentScore), 20, 70, 20, WHITE);
        DrawText(TextFormat("Top Score: %d", (int)(allTimeBestFitness/10)), 20, 95, 20, WHITE);
        
        DrawRectangleRounded(Rectangle{10, VIRTUAL_HEIGHT - 40, 450, 30}, 0.3f, 10, Fade(BLACK, 0.6f));
        DrawText(TextFormat("Speed: %dx | Left/Right Arrow | [F11] Full", speedMultiplier), 20, VIRTUAL_HEIGHT - 35, 20, WHITE);

        EndTextureMode();

        BeginDrawing();
        ClearBackground(BLACK);
        float scale = std::min((float)GetScreenWidth() / VIRTUAL_WIDTH, (float)GetScreenHeight() / VIRTUAL_HEIGHT);
        Rectangle sourceRec = { 0.0f, 0.0f, (float)target.texture.width, -(float)target.texture.height };
        Rectangle destRec = { 
            (GetScreenWidth() - ((float)VIRTUAL_WIDTH * scale)) * 0.5f, 
            (GetScreenHeight() - ((float)VIRTUAL_HEIGHT * scale)) * 0.5f, 
            (float)VIRTUAL_WIDTH * scale, 
            (float)VIRTUAL_HEIGHT * scale 
        };
        DrawTexturePro(target.texture, sourceRec, destRec, {0, 0}, 0.0f, WHITE);
        EndDrawing();
    }

    UnloadRenderTexture(target);
    UnloadTexture(bgTex);
    UnloadTexture(birdTex);
    UnloadTexture(pipeTex);

    CloseWindow();
    return 0;
}
