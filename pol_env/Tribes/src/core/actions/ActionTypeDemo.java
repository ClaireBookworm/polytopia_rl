package core.actions;

import core.Types;

import java.util.List;

/**
 * Demo class showing how to use ActionTypeUtils to get all possible action types
 * without generating specific action instances.
 */
public class ActionTypeDemo {
    
    public static void main(String[] args) {
        System.out.println("=== DEMO: Getting All Possible Action Types ===\n");
        
        // Get all possible action types
        List<Types.ACTION> allActions = ActionTypeUtils.getAllPossibleActionTypes();
        System.out.println("Total possible action types: " + allActions.size());
        System.out.println("All actions: " + allActions);
        System.out.println();
        
        // Get actions by category
        List<Types.ACTION> cityActions = ActionTypeUtils.getCityActionTypes();
        System.out.println("City actions (" + cityActions.size() + "): " + cityActions);
        
        List<Types.ACTION> tribeActions = ActionTypeUtils.getTribeActionTypes();
        System.out.println("Tribe actions (" + tribeActions.size() + "): " + tribeActions);
        
        List<Types.ACTION> unitActions = ActionTypeUtils.getUnitActionTypes();
        System.out.println("Unit actions (" + unitActions.size() + "): " + unitActions);
        
        List<Types.ACTION> otherActions = ActionTypeUtils.getOtherActionTypes();
        System.out.println("Other actions (" + otherActions.size() + "): " + otherActions);
        System.out.println();
        
        // Get actions that require specific technologies
        List<Types.ACTION> actionsRequiringChivalry = ActionTypeUtils.getActionTypesRequiringTech(Types.TECHNOLOGY.CHIVALRY);
        System.out.println("Actions requiring CHIVALRY: " + actionsRequiringChivalry);
        
        List<Types.ACTION> actionsRequiringSailing = ActionTypeUtils.getActionTypesRequiringTech(Types.TECHNOLOGY.SAILING);
        System.out.println("Actions requiring SAILING: " + actionsRequiringSailing);
        System.out.println();
        
        // Get actions without technology requirements
        List<Types.ACTION> noTechActions = ActionTypeUtils.getActionTypesWithoutTechRequirement();
        System.out.println("Actions without tech requirements (" + noTechActions.size() + "): " + noTechActions);
        System.out.println();
        
        // Get counts by category
        int[] counts = ActionTypeUtils.getActionTypeCountsByCategory();
        System.out.println("Action counts by category:");
        System.out.println("- City: " + counts[0]);
        System.out.println("- Tribe: " + counts[1]);
        System.out.println("- Unit: " + counts[2]);
        System.out.println("- Other: " + counts[3]);
        System.out.println();
        
        // Print detailed summary
        ActionTypeUtils.printActionTypeSummary();
    }
}
