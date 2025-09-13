package core.game;

import core.Types;
import core.actions.Action;
import core.actors.City;
import core.actors.Tribe;
import core.actors.units.Battleship;
import core.actors.units.Boat;
import core.actors.units.Ship;
import core.actors.units.Unit;
import org.json.JSONArray;
import org.json.JSONObject;
import gui.GUI;
import gui.WindowInput;
import players.ActionController;
import players.KeyController;
import players.Agent;
import players.DoNothingAgent;

import java.util.ArrayList;
import java.util.Random;

/**
 * Minimal bridge for Python interop. Exposes init, list-actions, step-by-index, observation JSON, and status.
 */
public class PythonEnv {

    private GameState gs;
    private Random rnd;
    private GUI gui;
    private Game viewerGame;

    /**
     * Initialize from a CSV level file path (relative to CWD or absolute), seed, and game mode.
     */
    public void initFromLevel(String filename, long seed, Types.GAME_MODE gameMode) {
        this.rnd = new Random(seed);
        this.gs = new GameState(rnd, gameMode);
        this.gs.init(filename);

        // Start the turn for tribe 0, mirroring Game.processTurn preconditions
        Tribe t0 = gs.getTribe(0);
        gs.initTurn(t0);
        gs.computePlayerActions(t0);
    }

    public int getActiveTribeID() {
        return gs.getActiveTribeID();
    }

    public int getTick() {
        return gs.getTick();
    }

    public boolean isDone() {
        return gs.isGameOver();
    }

    public String getGameMode() {
        return gs.getGameMode().toString();
    }

    public int[] getScores() {
        Tribe[] tribes = gs.getTribes();
        int[] scores = new int[tribes.length];
        for (int i = 0; i < tribes.length; i++) scores[i] = tribes[i].getScore();
        return scores;
    }

    /**
     * Returns each available action encoded as a JSON string: {idx, type, repr}
     */
    public java.util.List<String> listActionsJson() {
        ArrayList<Action> acts = gs.getAllAvailableActions();
        ArrayList<String> out = new ArrayList<>(acts.size());
        for (int i = 0; i < acts.size(); i++) {
            Action a = acts.get(i);
            JSONObject jo = new JSONObject();
            jo.put("idx", i);
            jo.put("type", a.getActionType().toString());
            jo.put("repr", a.toString());
            out.add(jo.toString());
        }
        return out;
    }

    public int actionCount() {
        return gs.getAllAvailableActions().size();
    }

    /**
     * Applies the action by its index in the last listActionsJson() call and recomputes actions.
     */
    public void stepByIndex(int idx) {
        ArrayList<Action> acts = gs.getAllAvailableActions();
        if (idx < 0 || idx >= acts.size()) throw new IllegalArgumentException("Invalid action index: " + idx);
        gs.advance(acts.get(idx), true);
    }

    /**
     * Returns an observation JSON similar to GameSaver output but in-memory.
     */
    public String observationJson() {
        JSONObject game = new JSONObject();

        // Board INFO (2D arrays)
        JSONObject board = new JSONObject();
        JSONArray terrain2D = new JSONArray();
        JSONArray resource2D = new JSONArray();
        JSONArray unit2D = new JSONArray();
        JSONArray city2D = new JSONArray();
        JSONArray building2D = new JSONArray();
        JSONArray network2D = new JSONArray();

        Board b = gs.getBoard();
        for (int i = 0; i < b.getSize(); i++) {
            JSONArray terrain = new JSONArray();
            JSONArray resource = new JSONArray();
            JSONArray units = new JSONArray();
            JSONArray cities = new JSONArray();
            JSONArray buildings = new JSONArray();
            JSONArray networks = new JSONArray();

            for (int j = 0; j < b.getSize(); j++) {
                terrain.put(b.getTerrainAt(i, j).getKey());
                resource.put(b.getResourceAt(i, j) != null ? b.getResourceAt(i, j).getKey() : -1);
                units.put(b.getUnitIDAt(i, j));
                cities.put(b.getCityIdAt(i, j));
                buildings.put(b.getBuildingAt(i, j) != null ? b.getBuildingAt(i, j).getKey() : -1);
                networks.put(b.getNetworkTilesAt(i, j));
            }

            terrain2D.put(terrain);
            resource2D.put(resource);
            unit2D.put(units);
            city2D.put(cities);
            building2D.put(buildings);
            network2D.put(networks);
        }

        // Unit INFO
        JSONObject unit = new JSONObject();
        for (Unit u : getAllUnits(b)) {
            JSONObject uInfo = new JSONObject();
            uInfo.put("type", u.getType().getKey());
            if (u.getType() == Types.UNIT.BOAT) {
                uInfo.put("baseLandType", ((Boat) u).getBaseLandUnit().getKey());
            } else if (u.getType() == Types.UNIT.SHIP) {
                uInfo.put("baseLandType", ((Ship) u).getBaseLandUnit().getKey());
            } else if (u.getType() == Types.UNIT.BATTLESHIP) {
                uInfo.put("baseLandType", ((Battleship) u).getBaseLandUnit().getKey());
            }
            uInfo.put("x", u.getPosition().x);
            uInfo.put("y", u.getPosition().y);
            uInfo.put("kill", u.getKills());
            uInfo.put("isVeteran", u.isVeteran());
            uInfo.put("cityID", u.getCityId());
            uInfo.put("tribeId", u.getTribeId());
            uInfo.put("currentHP", u.getCurrentHP());
            unit.put(String.valueOf(u.getActorId()), uInfo);
        }

        // City INFO
        JSONObject city = new JSONObject();
        for (City c : getAllCities(b)) {
            JSONObject cInfo = new JSONObject();
            cInfo.put("x", c.getPosition().x);
            cInfo.put("y", c.getPosition().y);
            cInfo.put("tribeID", c.getTribeId());
            cInfo.put("population_need", c.getPopulation_need());
            cInfo.put("bound", c.getBound());
            cInfo.put("level", c.getLevel());
            cInfo.put("isCapital", c.isCapital());
            cInfo.put("population", c.getPopulation());
            cInfo.put("production", c.getProduction());
            cInfo.put("hasWalls", c.hasWalls());
            cInfo.put("pointsWorth", c.getPointsWorth());
            JSONArray buildingList = new JSONArray();
            for (core.actors.Building bld : c.getBuildings()) {
                JSONObject bInfo = new JSONObject();
                bInfo.put("x", bld.position.x);
                bInfo.put("y", bld.position.y);
                bInfo.put("type", bld.type.getKey());
                if (bld.type.isTemple()) {
                    core.actors.Temple t = (core.actors.Temple) bld;
                    bInfo.put("level", t.getLevel());
                    bInfo.put("turnsToScore", t.getTurnsToScore());
                }
                buildingList.put(bInfo);
            }
            cInfo.put("buildings", buildingList);
            cInfo.put("units", c.getUnitsID());
            city.put(String.valueOf(c.getActorId()), cInfo);
        }

        // Tribes INFO (subset)
        JSONObject tribesINFO = new JSONObject();
        Tribe[] tribes = gs.getTribes();
        for (Tribe t : tribes) {
            JSONObject tribeInfo = new JSONObject();
            tribeInfo.put("citiesID", t.getCitiesID());
            tribeInfo.put("capitalID", t.getCapitalID());
            tribeInfo.put("type", t.getType().getKey());
            tribeInfo.put("star", t.getStars());
            tribeInfo.put("winner", t.getWinner().getKey());
            tribeInfo.put("score", t.getScore());
            tribeInfo.put("extraUnits", t.getExtraUnits());
            tribeInfo.put("nKills", t.getnKills());
            tribeInfo.put("nPacifistCount", t.getnPacifistCount());
            tribesINFO.put(String.valueOf(t.getActorId()), tribeInfo);
        }

        board.put("terrain", terrain2D);
        board.put("resource", resource2D);
        board.put("unitID", unit2D);
        board.put("cityID", city2D);
        board.put("network", network2D);
        board.put("building", building2D);
        board.put("actorIDcounter", b.getActorIDcounter());

        game.put("board", board);
        game.put("unit", unit);
        game.put("city", city);
        game.put("tribes", tribesINFO);
        game.put("tick", gs.getTick());
        game.put("gameIsOver", gs.isGameOver());
        game.put("activeTribeID", gs.getActiveTribeID());
        game.put("gameMode", gs.getGameMode().getKey());

        return game.toString();
    }

    /**
     * Opens a Java GUI window using the existing viewer components. Safe to call multiple times.
     */
    public void openGui() {
        if (gui != null) return;

        // Build a minimal Game just for GUI plumbing (players, pause state, etc.)
        viewerGame = new Game();
        int n = gs.getTribes().length;
        java.util.ArrayList<Agent> players = new java.util.ArrayList<>();
        long agentSeed = 0L;
        for (int i = 0; i < n; i++) players.add(new DoNothingAgent(agentSeed));

        // Initialize viewerGame with a generated level matching current tribes to set up players array
        // We won't use viewerGame's internal state for rendering; GUI.update uses the GameState we pass.
        Types.TRIBE[] tribes = new Types.TRIBE[n];
        for (int i = 0; i < n; i++) tribes[i] = gs.getTribe(i).getType();
        long levelSeed = 0L;
        viewerGame.init(players, levelSeed, tribes, 0L, gs.getGameMode());

        KeyController ki = new KeyController(true);
        WindowInput wi = new WindowInput();
        ActionController ac = new ActionController();
        gui = new GUI(viewerGame, "Tribes Viewer", ki, wi, ac, false);

        // Force zoomed out so the whole board fits initially
        try {
            int boardSize = gs.getBoard().getSize();
            int viewSize = core.Constants.GUI_GAME_VIEW_SIZE > 0 ? core.Constants.GUI_GAME_VIEW_SIZE : 600;
            int cell = Math.max(2, viewSize / Math.max(1, boardSize));
            core.Constants.CELL_SIZE = cell;
            gui.repaint();
        } catch (Exception ignored) {}
    }

    /**
     * Renders the current GameState in the Java GUI window; call openGui() beforehand.
     */
    public void renderGui() {
        if (gui == null) openGui();
        gui.update(gs.copy(-1), null);
    }

    /**
     * Closes the Java GUI window if open.
     */
    public void closeGui() {
        if (gui != null) {
            gui.dispose();
            gui = null;
            viewerGame = null;
        }
    }

    private static ArrayList<City> getAllCities(Board board) {
        Tribe[] tribes = board.getTribes();
        ArrayList<City> cityActors = new ArrayList<>();
        for (Tribe t : tribes) {
            ArrayList<Integer> cities = t.getCitiesID();
            for (Integer cityId : cities) {
                cityActors.add((City) board.getActor(cityId));
            }
        }
        return cityActors;
    }

    private static ArrayList<Unit> getAllUnits(Board board) {
        Tribe[] tribes = board.getTribes();
        ArrayList<Unit> unitActors = new ArrayList<>();
        for (Tribe t : tribes) {
            ArrayList<Integer> cities = t.getCitiesID();
            for (Integer cityId : cities) {
                City c = (City) board.getActor(cityId);
                for (Integer unitId : c.getUnitsID()) {
                    Unit unit = (Unit) board.getActor(unitId);
                    unitActors.add(unit);
                }
            }
            for (Integer unitId : t.getExtraUnits()) {
                Unit unit = (Unit) board.getActor(unitId);
                unitActors.add(unit);
            }
        }
        return unitActors;
    }
}


