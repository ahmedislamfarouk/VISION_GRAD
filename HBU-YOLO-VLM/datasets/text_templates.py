"""
Text Templates for Disaster Scene Captioning
"""


class DisasterTemplates:
    """Templates for generating disaster scene captions"""
    
    def __init__(self):
        self.templates = {
            'general': [
                "This disaster scene shows {objects}.",
                "The aerial view reveals {objects} in the affected area.",
                "Damage assessment identifies {objects}.",
                "UAV survey detected {objects}."
            ],
            'building_damage': [
                "Building damage level: {damage_level}. {objects} visible.",
                "Structural assessment: {damage_level}. Surrounding objects include {objects}.",
                "Building status: {damage_level}. Nearby: {objects}."
            ],
            'flood': [
                "Flood-affected area with {objects}.",
                "Water levels indicate severe flooding. {objects} partially submerged.",
                "Flood damage assessment: {objects} affected."
            ],
            'earthquake': [
                "Earthquake damage: {objects} affected.",
                "Seismic impact visible on {objects}.",
                "Post-earthquake assessment shows {objects}."
            ],
            'fire': [
                "Fire damage to {objects}.",
                "Burn area contains {objects}.",
                "Fire impact assessment: {objects} affected."
            ]
        }
    
    def format_counts(self, class_counts: dict) -> str:
        """Format object counts into caption"""
        if not class_counts:
            return "No objects detected in this disaster scene."
        
        # Build object list
        objects = []
        for class_name, count in class_counts.items():
            if count > 0:
                objects.append(f"{count} {class_name}{'s' if count > 1 else ''}")
        
        if not objects:
            return "No objects detected."
        
        # Format with template
        objects_str = ", ".join(objects[:-1]) + f" and {objects[-1]}" if len(objects) > 1 else objects[0]
        
        template = self.templates['general'][0]
        return template.format(objects=objects_str)
    
    def format_damage_assessment(
        self,
        class_counts: dict,
        damage_level: str = "moderate"
    ) -> str:
        """Format damage assessment caption"""
        objects_str = self.format_counts(class_counts)
        
        template = self.templates['building_damage'][0]
        return template.format(
            damage_level=damage_level,
            objects=objects_str
        )


def get_disaster_templates() -> DisasterTemplates:
    """Get disaster templates instance"""
    return DisasterTemplates()
