// smartPaGateway.ts — patient only
import { Request, Response } from "express";
import { CprPlusEntityBridge } from "./cprPlusEntityBridge";
import { PatientBridge } from "./CprPlusPatientBridge";

const patient = new PatientBridge();
console.log("patient bridge:", patient?.constructor?.name, "| create:", typeof patient?.create);

const create = (bridge: CprPlusEntityBridge) => async (payload: any) => ({
  status: 201,
  body: { success: true, ...(await bridge.create(payload)) },
});

const routes: Record<string, (payload: any) => Promise<{ status: number; body: any }>> = {
  "patient:create": create(patient),

  "patient:lookup": async (payload) => {
    const p = payload ?? {};
    if (p.mrn) {
      const env = await patient.getByMrn(String(p.mrn));
      return env
        ? { status: 200, body: env }
        : { status: 404, body: { error: "Patient not found" } };
    }
    const { firstName, lastName, dob } = p;
    if (!firstName || !lastName || !dob) {
      return { status: 400, body: { error: "Provide mrn, or firstName + lastName + dob" } };
    }
    const list = await patient.findByNameDob(firstName, lastName, dob);
    return list.length
      ? { status: 200, body: { count: list.length, patients: list } }
      : { status: 404, body: { error: "No matching patients found" } };
  },
};

export async function smartPaGateway(req: Request, res: Response): Promise<void> {
  try {
    const { entity, operation, payload } = req.body ?? {};
    if (!entity || !operation) {
      res.status(400).json({ error: "entity and operation are required" });
      return;
    }
    const route = routes[`${entity}:${operation}`];
    if (!route) {
      res.status(400).json({ error: `Unsupported: ${entity}:${operation}` });
      return;
    }
    const { status, body } = await route(payload);
    res.status(status).json(body);
  } catch (err: any) {
    const status = typeof err?.status === "number" ? err.status : 500;
    if (status === 500) console.error("smartPaGateway failed:", err);
    res.status(status).json({ error: status === 500 ? "Request failed" : err.message });
  }
}