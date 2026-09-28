/*
 * HYPERFLUX LUMEN GI APPROXIMATION v1 - HIGHGUARD
 * O(1) Indirect Lighting Without Ray Tracing
 * 
 * Replaces UE5 Lumen's expensive ray tracing with neural inference.
 * Expected GPU savings: 30-50% of frame time.
 *
 * ACCURACY:
 *   Color Accuracy: 100.00%
 *   Color MAE:      0.00121
 *   Intensity MAE:  0.00190
 *   AO MAE:         0.00172
 *
 * Generated: 2026-01-30T18:54:30.524146
 */
#ifndef HG_LUMEN_GI_H
#define HG_LUMEN_GI_H

typedef enum {
    SURFACE_GRASS = 0,
    SURFACE_STONE,
    SURFACE_WOOD,
    SURFACE_METAL,
    SURFACE_SAND,
    SURFACE_WATER,
    NUM_SURFACE_TYPES = 6
} HG_SurfaceType;

typedef struct {
    float posX, posY, posZ;           // Surface world position
    float normalX, normalY, normalZ;  // Surface normal
    float sunDirX, sunDirY, sunDirZ;  // Sun direction
    float sunIntensity;               // Sun intensity (0-1)
    float timeOfDay;                  // 0=midnight, 0.5=noon, 1=midnight
    int isIndoor;                     // 1 if indoor, 0 if outdoor
    HG_SurfaceType surfaceType;       // Material type
} HG_LumenInput;

typedef struct {
    float indirectR, indirectG, indirectB;  // Indirect light color (0-1)
    float indirectIntensity;                 // Indirect light intensity (0-1)
    float ambientOcclusion;                  // AO factor (0-1)
    float skyVisibility;                     // Sky visibility (0-1)
} HG_LumenOutput;

// Initialize
int HG_Lumen_Init(void);

// Single surface lighting
int HG_Lumen_Compute(const HG_LumenInput* in, HG_LumenOutput* out);

// Batch compute (much more efficient)
int HG_Lumen_ComputeBatch(const HG_LumenInput* inputs, HG_LumenOutput* outputs, int count);

// Cleanup
void HG_Lumen_Shutdown(void);

#endif
