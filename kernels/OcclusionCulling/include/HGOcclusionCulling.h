/*
 * HYPERFLUX OCCLUSION CULLING v2 - HIGHGUARD
 * O(1) Visibility Prediction Without GPU Stalls
 *
 * ACCURACY:
 *   Visibility:        87.04%
 *   Recall:            99.75%
 *   False Negative:    0.25%
 *
 * Generated: 2026-01-30T18:49:44.746137
 */
#ifndef HG_OCCLUSION_CULLING_H
#define HG_OCCLUSION_CULLING_H

typedef enum {
    OCCLUDER_NONE = 0,
    OCCLUDER_TERRAIN_SMALL,
    OCCLUDER_TERRAIN_LARGE,
    OCCLUDER_WALL_THIN,
    OCCLUDER_WALL_THICK,
    OCCLUDER_STRUCTURE,
    NUM_OCCLUDER_TYPES = 6
} HG_OccluderType;

typedef struct {
    float cameraPosX, cameraPosY, cameraPosZ;
    float cameraForwardX, cameraForwardY, cameraForwardZ;
    float cameraFOV;
} HG_CameraInfo;

typedef struct {
    float posX, posY, posZ;
    float boundingRadius;
} HG_ObjectBounds;

typedef struct {
    float posX, posY, posZ;
    float size;
    HG_OccluderType type;
} HG_OccluderInfo;

typedef struct {
    int isVisible;
    float confidence;
} HG_VisibilityResult;

int HG_Occlusion_Init(void);
int HG_Occlusion_Test(const HG_CameraInfo* camera,
                      const HG_ObjectBounds* object,
                      const HG_OccluderInfo* occluder,
                      HG_VisibilityResult* result);
int HG_Occlusion_TestBatch(const HG_CameraInfo* camera,
                           const HG_ObjectBounds* objects,
                           const HG_OccluderInfo* occluders,
                           HG_VisibilityResult* results,
                           int count);
void HG_Occlusion_Shutdown(void);

#endif
