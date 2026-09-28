/*
 * HYPERFLUX LOD SELECTION v1 - HIGHGUARD
 * O(1) Batch Level-of-Detail Prediction
 * 
 * Replaces per-object LOD calculations with neural batch inference.
 * Includes predictive LOD based on camera velocity.
 *
 * ACCURACY:
 *   LOD Exact:     99.39%
 *   LOD Within-1:  99.89%
 *   Cull:          99.79%
 *
 * Generated: 2026-01-30T18:41:06.293593
 */
#ifndef HG_LOD_SELECTION_H
#define HG_LOD_SELECTION_H

typedef enum {
    IMPORTANCE_HERO = 0,      // Wardens, mounts
    IMPORTANCE_CRITICAL,      // Weapons, Shieldbreaker, generators
    IMPORTANCE_STRUCTURE,     // Base walls, siege tower
    IMPORTANCE_ENVIRONMENT,   // Rocks, trees
    IMPORTANCE_DETAIL,        // Small props, debris
    NUM_IMPORTANCE_LEVELS = 5
} HG_ObjectImportance;

typedef struct {
    float cameraPosX, cameraPosY, cameraPosZ;
    float cameraForwardX, cameraForwardY, cameraForwardZ;
    float cameraFOV;          // Degrees (typically 90-110)
    float cameraVelX, cameraVelY, cameraVelZ;
} HG_CameraState;

typedef struct {
    float posX, posY, posZ;
    float boundingRadius;     // Bounding sphere radius in meters
    HG_ObjectImportance importance;
} HG_ObjectInfo;

typedef struct {
    int lodLevel;             // 0 (highest) to 4 (lowest)
    float blendFactor;        // 0-1 for smooth LOD transitions
    int shouldCull;           // 1 = too far/small, don't render
} HG_LODResult;

// Initialize
int HG_LOD_Init(void);

// Single object LOD selection
int HG_LOD_Select(const HG_CameraState* camera, const HG_ObjectInfo* obj, HG_LODResult* result);

// Batch LOD selection (much more efficient for many objects)
int HG_LOD_SelectBatch(const HG_CameraState* camera, const HG_ObjectInfo* objects, 
                       HG_LODResult* results, int count);

// Cleanup
void HG_LOD_Shutdown(void);

#endif
