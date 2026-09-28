/*
 * HYPERFLUX BALLISTICS KERNEL - HIGHGUARD EDITION
 * O(1) Bullet Trajectory Prediction
 * 
 * Drop Accuracy: 99.76%
 * Damage Accuracy: 87.29%
 * 
 * Copyright (c) 2026 HyperFlux Neural Dynamics
 */

#ifndef HYPERFLUX_HIGHGUARD_BALLISTICS_H
#define HYPERFLUX_HIGHGUARD_BALLISTICS_H

// Weapon IDs
typedef enum {
    WEAPON_RANGER = 0,      // Sniper
    WEAPON_KRAKEN = 1,      // Pump Shotgun
    WEAPON_PALADIN = 2,     // Auto Shotgun
    WEAPON_CORSAIR = 3,     // Burst SMG
    WEAPON_VIPER = 4,       // SMG
    WEAPON_DYNASTY = 5,     // AR
    WEAPON_VANGUARD = 6,    // AR
    WEAPON_SABER = 7,       // Burst AR
    WEAPON_LONGHORN = 8,    // Revolver
    WEAPON_BIGRIG = 9,      // LMG
    NUM_WEAPONS = 10
} HG_WeaponID;

// Rarity IDs
typedef enum {
    RARITY_WHITE = 0,       // Common
    RARITY_BLUE = 1,        // Uncommon
    RARITY_PURPLE = 2,      // Rare
    RARITY_GOLD = 3,        // Epic
    RARITY_RED = 4,         // Legendary
    NUM_RARITIES = 5
} HG_RarityID;

// Warden IDs
typedef enum {
    WARDEN_ATTICUS = 0,     // Assault - Lightning
    WARDEN_SLADE = 1,       // Assault - Fire
    WARDEN_SCARLET = 2,     // Assault - Sand
    WARDEN_REDMANE = 3,     // Destruction - Beast
    WARDEN_KAI = 4,         // Defensive - Ice
    WARDEN_UNA = 5,         // Defensive - Spirits
    WARDEN_MARA = 6,        // Support - Soul
    WARDEN_CONDOR = 7,      // Recon - Bird
    NUM_WARDENS = 8
} HG_WardenID;

// Input structure
typedef struct {
    HG_WeaponID weapon;
    HG_RarityID rarity;
    HG_WardenID warden;
    float distance;         // meters
    int shotNumber;         // 1-based shot in spray
    int isADS;              // 0 or 1
    int isMoving;           // 0 or 1
    int redmaneEnraged;     // 0 or 1 (Redmane passive active)
} HG_BallisticsInput;

// Output structure
typedef struct {
    float dropOffset;       // meters (bullet drop)
    float flightTime;       // seconds
    float damage;           // damage points
    float spreadX;          // degrees
    float spreadY;          // degrees
} HG_BallisticsOutput;

// API Functions
int HG_Ballistics_Initialize(void);
void HG_Ballistics_Shutdown(void);
int HG_Ballistics_Predict(const HG_BallisticsInput* input, HG_BallisticsOutput* output);
const char* HG_Ballistics_GetVersion(void);

#endif // HYPERFLUX_HIGHGUARD_BALLISTICS_H
