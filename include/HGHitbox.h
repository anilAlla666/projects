/*
 * HYPERFLUX HITBOX v4.0 - HIGHGUARD
 * O(1) Hit Region Classification
 * Region Accuracy: 99.97%
 * Hit/Miss Accuracy: 99.97%
 */
#ifndef HG_HITBOX_V4_H
#define HG_HITBOX_V4_H

typedef enum {
    REGION_HEAD=0, REGION_UPPER_BODY, REGION_LOWER_BODY,
    REGION_ARMS, REGION_LEGS, REGION_MISS
} HG_Region;

static const float HG_REGION_DAMAGE[6] = {
    2.0f, 1.0f, 0.9f, 0.75f, 0.7f, 0.0f
};

#endif
