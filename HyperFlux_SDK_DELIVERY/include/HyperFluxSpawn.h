/*
 * HyperFlux Spawn Selection Kernel v4.0
 * Copyright (c) 2026 HyperFlux Neural Dynamics
 */

#ifndef HYPERFLUX_SPAWN_H
#define HYPERFLUX_SPAWN_H

#include <cstdint>

namespace HyperFlux { namespace Spawn {

constexpr const char* VERSION = "4.0.0";
constexpr int GRID_RES = 16;
constexpr int MAX_SPAWNS = 30;
constexpr int MAX_ENEMIES = 6;
constexpr int MAX_TEAMMATES = 5;
constexpr float MAP_SIZE = 60.0f;
constexpr int TIER_EXCELLENT = 3;
constexpr int TIER_GOOD = 9;
constexpr int TIER_ACCEPTABLE = 18;

enum class SpawnTier : uint8_t { Excellent=0, Good=1, Acceptable=2, Bad=3 };
enum class ErrorCode : uint32_t { Success=0, InvalidLicense=1, NotInitialized=2 };

struct Vec3 { float x,y,z; Vec3():x(0),y(0),z(0){} Vec3(float a,float b,float c):x(a),y(b),z(c){} };
struct Enemy { Vec3 position; bool active; Enemy():active(false){} };
struct Teammate { Vec3 position; bool active; Teammate():active(false){} };
struct ObjectiveState { int8_t a,b,c; ObjectiveState():a(0),b(0),c(0){} };

struct GameState {
    Enemy enemies[MAX_ENEMIES];
    Teammate teammates[MAX_TEAMMATES];
    uint8_t numEnemies, numTeammates;
    ObjectiveState objectives;
    float gameTime, scoreDiff;
    GameState():numEnemies(0),numTeammates(0),gameTime(0),scoreDiff(0){}
};

struct SpawnResult {
    uint8_t bestSpawnIdx;
    float scores[MAX_SPAWNS];
    uint8_t rankings[MAX_SPAWNS];
    SpawnTier bestTier;
    float confidence;
    SpawnResult():bestSpawnIdx(0),bestTier(SpawnTier::Good),confidence(0){}
};

class Engine {
public:
    Engine();
    ~Engine();
    bool Initialize(const char* licenseKey);
    bool IsReady() const;
    ErrorCode SelectSpawn(const GameState& state, SpawnResult& result) const;
    Vec3 GetSpawnPosition(uint8_t idx) const;
    static SpawnTier GetTierForRank(uint8_t rank);
private:
    class Impl; Impl* impl_;
    Engine(const Engine&)=delete;
    Engine& operator=(const Engine&)=delete;
};

const char* SpawnTierToString(SpawnTier tier);

}} // namespace
#endif
