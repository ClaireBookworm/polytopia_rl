package core.actions;

import core.Types;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Utility class for working with action types without generating specific action instances.
 * This provides static methods to get all possible action types and categorize them.
 */
public class ActionTypeUtils {

    /**
     * Gets all possible action types defined in the ACTION enum.
     * @return List of all possible action types
     */
    public static List<Types.ACTION> getAllPossibleActionTypes() {
        return Arrays.asList(Types.ACTION.values());
    }

    /**
     * Gets all city action types.
     * @return List of city action types
     */
    public static List<Types.ACTION> getCityActionTypes() {
        List<Types.ACTION> cityActions = new ArrayList<>();
        for (Types.ACTION action : Types.ACTION.values()) {
            if (isCityAction(action)) {
                cityActions.add(action);
            }
        }
        return cityActions;
    }

    /**
     * Gets all tribe action types.
     * @return List of tribe action types
     */
    public static List<Types.ACTION> getTribeActionTypes() {
        List<Types.ACTION> tribeActions = new ArrayList<>();
        for (Types.ACTION action : Types.ACTION.values()) {
            if (isTribeAction(action)) {
                tribeActions.add(action);
            }
        }
        return tribeActions;
    }

    /**
     * Gets all unit action types.
     * @return List of unit action types
     */
    public static List<Types.ACTION> getUnitActionTypes() {
        List<Types.ACTION> unitActions = new ArrayList<>();
        for (Types.ACTION action : Types.ACTION.values()) {
            if (isUnitAction(action)) {
                unitActions.add(action);
            }
        }
        return unitActions;
    }

    /**
     * Gets all other action types (not city, tribe, or unit).
     * @return List of other action types
     */
    public static List<Types.ACTION> getOtherActionTypes() {
        List<Types.ACTION> otherActions = new ArrayList<>();
        for (Types.ACTION action : Types.ACTION.values()) {
            if (!isCityAction(action) && !isTribeAction(action) && !isUnitAction(action)) {
                otherActions.add(action);
            }
        }
        return otherActions;
    }

    /**
     * Checks if an action type is a city action.
     * @param action The action type to check
     * @return true if it's a city action
     */
    public static boolean isCityAction(Types.ACTION action) {
        switch (action) {
            case BUILD:
            case BURN_FOREST:
            case CLEAR_FOREST:
            case DESTROY:
            case GROW_FOREST:
            case LEVEL_UP:
            case RESOURCE_GATHERING:
            case SPAWN:
                return true;
            default:
                return false;
        }
    }

    /**
     * Checks if an action type is a tribe action.
     * @param action The action type to check
     * @return true if it's a tribe action
     */
    public static boolean isTribeAction(Types.ACTION action) {
        switch (action) {
            case BUILD_ROAD:
            case END_TURN:
            case RESEARCH_TECH:
            case DECLARE_WAR:
            case SEND_STARS:
                return true;
            default:
                return false;
        }
    }

    /**
     * Checks if an action type is a unit action.
     * @param action The action type to check
     * @return true if it's a unit action
     */
    public static boolean isUnitAction(Types.ACTION action) {
        switch (action) {
            case ATTACK:
            case CAPTURE:
            case CONVERT:
            case DISBAND:
            case EXAMINE:
            case HEAL_OTHERS:
            case MAKE_VETERAN:
            case MOVE:
            case RECOVER:
                return true;
            default:
                return false;
        }
    }

    /**
     * Gets action types that require a specific technology.
     * @param tech The technology to check for
     * @return List of action types that require this technology
     */
    public static List<Types.ACTION> getActionTypesRequiringTech(Types.TECHNOLOGY tech) {
        List<Types.ACTION> actions = new ArrayList<>();
        for (Types.ACTION action : Types.ACTION.values()) {
            if (action.getTechnologyRequirement() == tech) {
                actions.add(action);
            }
        }
        return actions;
    }

    /**
     * Gets action types that don't require any technology.
     * @return List of action types that don't require technology
     */
    public static List<Types.ACTION> getActionTypesWithoutTechRequirement() {
        List<Types.ACTION> actions = new ArrayList<>();
        for (Types.ACTION action : Types.ACTION.values()) {
            if (action.getTechnologyRequirement() == null) {
                actions.add(action);
            }
        }
        return actions;
    }

    /**
     * Gets the count of all possible action types.
     * @return Total number of action types
     */
    public static int getTotalActionTypeCount() {
        return Types.ACTION.values().length;
    }

    /**
     * Gets the count of action types by category.
     * @return Array with counts [city, tribe, unit, other]
     */
    public static int[] getActionTypeCountsByCategory() {
        int cityCount = getCityActionTypes().size();
        int tribeCount = getTribeActionTypes().size();
        int unitCount = getUnitActionTypes().size();
        int otherCount = getOtherActionTypes().size();
        return new int[]{cityCount, tribeCount, unitCount, otherCount};
    }

    /**
     * Prints a summary of all action types organized by category.
     */
    public static void printActionTypeSummary() {
        System.out.println("=== ACTION TYPE SUMMARY ===");
        System.out.println("Total action types: " + getTotalActionTypeCount());
        
        int[] counts = getActionTypeCountsByCategory();
        System.out.println("City actions: " + counts[0]);
        System.out.println("Tribe actions: " + counts[1]);
        System.out.println("Unit actions: " + counts[2]);
        System.out.println("Other actions: " + counts[3]);
        
        System.out.println("\n=== CITY ACTIONS ===");
        for (Types.ACTION action : getCityActionTypes()) {
            System.out.println("- " + action + (action.getTechnologyRequirement() != null ? 
                " (requires " + action.getTechnologyRequirement() + ")" : ""));
        }
        
        System.out.println("\n=== TRIBE ACTIONS ===");
        for (Types.ACTION action : getTribeActionTypes()) {
            System.out.println("- " + action + (action.getTechnologyRequirement() != null ? 
                " (requires " + action.getTechnologyRequirement() + ")" : ""));
        }
        
        System.out.println("\n=== UNIT ACTIONS ===");
        for (Types.ACTION action : getUnitActionTypes()) {
            System.out.println("- " + action + (action.getTechnologyRequirement() != null ? 
                " (requires " + action.getTechnologyRequirement() + ")" : ""));
        }
        
        System.out.println("\n=== OTHER ACTIONS ===");
        for (Types.ACTION action : getOtherActionTypes()) {
            System.out.println("- " + action + (action.getTechnologyRequirement() != null ? 
                " (requires " + action.getTechnologyRequirement() + ")" : ""));
        }
    }
}
