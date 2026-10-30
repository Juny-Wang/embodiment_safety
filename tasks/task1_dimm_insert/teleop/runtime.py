"""Isaac-only collector. No training, joint teleport, or source-layer saving."""
import json, importlib, importlib.util, time, hashlib
from pathlib import Path
from collections import deque
import numpy as np
from scipy.spatial.transform import Rotation
from tasks.task1_dimm_insert.env.data import Episode,verify_manifest,dump,clean
from tasks.task1_dimm_insert.env.actions import keyboard_action,apply_velocity,apply_soft_xy_attract

ROOT=Path("/mnt/data2/wjy_SERL/Embodied_Safety")
TASK=ROOT/"tasks/task1_dimm_insert"
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class CollectionPanel:
    _image_quality_hook=True
    _feedback_native_hook=True
    def __init__(self,show_ui=True):
        import omni.usd,omni.physx,omni.timeline,omni.kit.app,omni.ui as ui,carb.input
        from pxr import UsdPhysics,PhysxSchema
        self.cfg=json.loads((TASK/"configs/collection.json").read_text())
        self.manifest=verify_manifest(TASK/"scene/manifest.json")
        self.stage=omni.usd.get_context().get_stage()
        expected=(TASK/"scene/collection.usda").resolve()
        if Path(self.stage.GetRootLayer().realPath).resolve()!=expected:raise RuntimeError("Open "+str(expected)+" first")
        self.timeline=omni.timeline.get_timeline_interface()
        if self.timeline.is_playing():raise RuntimeError("Pause before loading collection panel")
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        self.closed=False;self.enabled=False;self.held=set();self.active=None;self.t=0.;self.clock=0.
        self.action=np.zeros(7);self.pending_obs=None;self.last_obs=None;self.pending_contacts=[]
        self.image_cache={};self.saved_frames={};self.violations=set();self.interval_peaks={}
        self.record_error="";self.last_saved_path="";self.record_started_t=0.;self.last_write_wall=0.
        self.sensor_errors=[];self.notice="Play -> INITIALIZE -> ENABLE -> RECORD"
        self.history=deque(maxlen=300);self.last_refresh=0.;self.subcommands=[];self.camera_errors=[]
        self.safety=json.loads((TASK/"configs/safety_source.json").read_text())
        self.calib=json.loads((TASK/"configs/geometry_source.json").read_text())
        self.assistance=None;self.assistance_state={};self.soft_attract_state={}
        if self.cfg.get("physics_mode")=="legacy_plan_d_assisted":
            self.assistance=load("_collection_baseline_assistance",TASK/"teleop/vendor/plan_d_runtime.py")
            self.assistance_state=self.assistance.apply_session_plan_d(self.stage,self.calib,self.safety)
            if not self.assistance_state.get("ok"):raise RuntimeError("Baseline assistance initialization failed")
        control=dict(self.safety["control"])
        control.update(translation_speed_m_s=self.cfg["action"]["translation_m_s"],
                       rotation_speed_rad_s=self.cfg["action"]["rotation_rad_s"],arm_joint_speed_rad_s=1.,
                       gripper_speed_m_s=self.cfg["action"]["gripper_m_s"],
                       tracking_error_stop_m=.10,ik_position_tolerance_m=.002,
                       ik_rotation_tolerance_rad=.05235987756,ik_max_evaluations=80)
        c=load("_collection_controller",TASK/"teleop/vendor/franka_teleop.py")
        self.controller=c.FrankaTeleop(dict(self.safety,control=control))
        self.bodies={};self.body_paths={}
        for role,path in self.cfg["bodies"].items():
            prim=self.stage.GetPrimAtPath(path)
            if not prim or not prim.HasAPI(UsdPhysics.RigidBodyAPI):raise RuntimeError("Missing rigid body "+path)
            self.body_paths[role]=path
        # Measurement schemas and timestep only. No drive retuning or geometry edits.
        for prim in self.stage.Traverse():
            if prim.IsA(UsdPhysics.Scene):
                PhysxSchema.PhysxSceneAPI.Apply(prim).CreateTimeStepsPerSecondAttr(self.cfg["physics_hz"])
            if prim.HasAPI(UsdPhysics.RigidBodyAPI) and any(str(prim.GetPath()).startswith(p) for p in ["/World/Robot","/World/MatchedSocket","/World/MatchedDIMM"]):
                PhysxSchema.PhysxContactReportAPI.Apply(prim).CreateThresholdAttr(0.)
        setup=load("_collection_cameras",TASK/"teleop/vendor/cameras_setup.py")
        self.camera_setup=setup.apply_session_cameras(self.stage,self.safety,self.calib)
        if not self.camera_setup.get("ok"):raise RuntimeError("Camera setup failed: "+repr(self.camera_setup))
        from isaacsim.sensors.camera import Camera
        self.cameras={}
        for key in self.cfg["required_cameras"]:
            cam=Camera(prim_path=self.camera_setup["paths"][key],name="collect_"+key,
                       resolution=tuple(self.cfg["image_resolution"]))
            cam.initialize();self.cameras[key]=cam
        self.physics_sub=omni.physx.get_physx_interface().subscribe_physics_step_events(self.physics_step)
        self.contact_sub=omni.physx.get_physx_simulation_interface().subscribe_contact_report_events(self.on_contacts)
        self.update_sub=omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(self.update)
        self.timeline_sub=self.timeline.get_timeline_event_stream().create_subscription_to_pop(self.timeline_event)
        self.input=carb.input.acquire_input_interface()
        self.key_sub=self.input.subscribe_to_input_events(self.on_input,order=-100001)
        self.ui=None
        feedback=load("_task1_safety_feedback",TASK/"teleop/safety_feedback.py")
        feedback.install(self,show_ui=show_ui)
        if show_ui:
            self.build_ui(ui)
            self.command(self.show_previews)
    def show_previews(self):
        if not hasattr(self,"preview_module"):
            self.preview_module=load("_task1_previews",TASK/"teleop/camera_previews.py")
        result=self.preview_module.show(self)
        self.notice="Three camera previews restored"
        return result
    def build_ui(self,ui):
        self.ui=ui.Window("Task1 DATA COLLECTION v002",width=610,height=510)
        with self.ui.frame:
            with ui.VStack(spacing=8):
                ui.Label("ASSISTED BASELINE COLLECTION | soft XY attract ON | mechanical success NOT certified",word_wrap=True)
                self.record_banner=ui.Label("NOT RECORDING",height=65,word_wrap=True,style={"font_size":20,"color":0xff40ccff})
                self.status=ui.Label(self.notice,word_wrap=True)
                with ui.HStack():
                    ui.Button("INITIALIZE",clicked_fn=lambda:self.command(self.initialize))
                    ui.Button("ENABLE",clicked_fn=lambda:self.command(self.enable))
                    ui.Button("DISABLE",clicked_fn=lambda:self.command(self.disable))
                    ui.Button("E-STOP",clicked_fn=self.estop)
                with ui.HStack():
                    ui.Button("RECORD",clicked_fn=lambda:self.command(self.start))
                    ui.Button("END: SUCCESS REQUEST",clicked_fn=lambda:self.command(lambda:self.finish("success_request")))
                    ui.Button("FAILED: DISCARD",clicked_fn=lambda:self.command(lambda:self.finish("failed")))
                ui.Button("SHOW / RESTORE 3 CAMERA VIEWS",clicked_fn=lambda:self.command(self.show_previews))
                ui.Button("RESET: end episode + reopen clean scene",clicked_fn=lambda:self.command(self.reset))
                ui.Label("Arrows: XY | PageUp/Down: Z | Shift: rotate | Ctrl: fine | [ / ]: gripper | Esc: stop\nDefault jog 0.32 m/s (~1.8x old); Ctrl fine 0.006 m/s. Soft XY snap near socket mouth = ASSIST only (not success).",word_wrap=True)
                self.values=ui.Label("Measured state: waiting",word_wrap=True)
                self.quality=ui.Label("Required: external + wrist left/right. No HUD or warning colours in recorded RGB.",word_wrap=True)
                ui.Label("Success request is saved for review; it is not a safe-demo acceptance. Source scene stays read-only.",word_wrap=True)
    def command(self,fn):
        try:fn()
        except Exception as e:self.notice=repr(e);print("[collection]",repr(e))
    def initialize(self):
        import omni.physics.tensors as tensors
        if self.active:raise RuntimeError("End episode before initialize")
        if not self.timeline.is_playing():raise RuntimeError("Play first")
        self.controller.initialize()
        self.sv=tensors.create_simulation_view("numpy");self.sv.set_subspace_roots("/")
        for role,path in self.body_paths.items():
            v=self.sv.create_rigid_body_view(path)
            if v.count!=1:raise RuntimeError("Expected single body: "+path)
            self.bodies[role]=v
        from pxr import UsdPhysics
        hand=[p for p in self.stage.Traverse() if str(p.GetPath()).startswith("/World/Robot/") and p.GetName()=="panda_hand" and p.HasAPI(UsdPhysics.RigidBodyAPI)]
        if len(hand)!=1:raise RuntimeError("Expected single physical hand")
        self.hand=self.sv.create_rigid_body_view(str(hand[0].GetPath()))
        a=self.hand.get_transforms()[0];pos,rot=self.controller.pose()
        if np.linalg.norm(a[:3]-pos)>.0005 or (Rotation.from_quat(a[3:]).inv()*Rotation.from_matrix(rot)).magnitude()>np.deg2rad(.5):
            self.controller.ready=False;raise RuntimeError("Physical hand / FK mismatch")
        self.enabled=False;self.history.clear();self.notice="Initialized. ENABLE; wait for stable frames before RECORD."
    def toggle(self):
        if not self.controller.ready:raise RuntimeError("Initialize first")
        if self.active:raise RuntimeError("End episode before disabling")
        if self.enabled:self.controller.stop("DISABLED");self.enabled=False;self.held.clear()
        else:self.controller.arm();self.enabled=True
        self.notice="Keyboard "+("ENABLED" if self.enabled else "DISABLED")
    def record_event(self,event,**fields):
        import os
        path=ROOT/"outputs/datasets/task1_dimm_insert/recording_events.jsonl"
        path.parent.mkdir(parents=True,exist_ok=True)
        row=dict(event=event,wall_time=time.time(),sim_time=self.t,**fields)
        with path.open("a") as f:
            f.write(json.dumps(clean(row),allow_nan=False)+"\n")
            f.flush();os.fsync(f.fileno())
    def enable(self):
        if not self.controller.ready:raise RuntimeError("INITIALIZE first")
        if not self.enabled:
            self.controller.arm();self.enabled=True
        self.notice="Keyboard ENABLED"
    def disable(self):
        if self.controller.ready:self.controller.stop("DISABLED")
        self.enabled=False;self.held.clear()
        self.notice="Keyboard DISABLED; recording continues if active"
    def start(self):
        if self.active:
            self.notice="Already recording: "+self.active.path.name
            return
        self.record_error=""
        self.record_event("start_requested",playing=self.timeline.is_playing(),
                          initialized=bool(self.controller.ready),keyboard_enabled=self.enabled)
        try:
            importlib.invalidate_caches()
            self._start()
            self.record_started_t=self.t
            self.last_write_wall=time.monotonic()
            self.record_event("recording_opened",path=str(self.active.path))
        except Exception as e:
            self.record_error=repr(e)
            if self.active:
                self.finish("interrupted","record_start_error:"+repr(e))
            self.record_event("start_failed",reason=repr(e))
            self.notice="NOT RECORDING: "+repr(e)
            raise
    def recording_status(self):
        if self.active:
            count=self.active.count
            stalled=self.timeline.is_playing() and time.monotonic()-self.last_write_wall>2.
            state="RECORDING STALLED" if stalled else ("RECORDING" if count else "WAITING FOR FIRST SAMPLE")
            return (state+" | %d saved transitions | %.1f sim s\n%s"%(count,max(0.,self.t-self.record_started_t),self.active.path),
                    0xff4040ff if stalled else (0xff60cc60 if count else 0xff40ccff))
        if self.record_error:return ("NOT RECORDING - START FAILED\n"+self.record_error,0xff4040ff)
        if self.last_saved_path:return ("NOT RECORDING | last saved:\n"+self.last_saved_path,0xff40ccff)
        return ("NOT RECORDING | Play -> INITIALIZE -> RECORD\nKeyboard ENABLE is separate.",0xff40ccff)

    def _start(self):
        if not self.timeline.is_playing():raise RuntimeError("Simulation paused: click Play")
        if not self.controller.ready or not self.bodies:raise RuntimeError("Sensors/control not initialized: click INITIALIZE")
        if self.active:raise RuntimeError("Already recording")
        self.manifest=verify_manifest(TASK/"scene/manifest.json")
        recent_speeds=[v for timestamp,v in self.history if self.t-timestamp<=.5]
        initial_stability=dict(observed_window_s=min(.5,self.t-self.history[0][0]) if self.history else 0.,
                               max_body_speed_m_s=max(recent_speeds,default=None),
                               settled=len(self.history)>1 and self.t-self.history[0][0]>=.5 and max(recent_speeds,default=0.)<=.01)
        self.violations=set();self.interval_peaks={};self.saved_frames={};self.sensor_errors=[]
        meta=dict(initial_stability=initial_stability,keyboard_enabled_at_start=self.enabled,scene_manifest=self.manifest,control_dt=1/self.cfg["control_hz"],
                  physics_dt=1/self.cfg["physics_hz"],required_cameras=self.cfg["required_cameras"],
                  max_image_age_s=self.cfg["max_image_age_s"],configuration=self.cfg,
                  action_contract="7D normalized WORLD velocity: xyz,rotation-vector xyz,gripper width rate",
                  physics_mode=self.cfg.get("physics_mode","authored_scene"),
                  runtime_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  feedback_sha256=hashlib.sha256((TASK/"teleop/safety_feedback.py").read_bytes()).hexdigest(),
                  grasp_region_source="configs/geometry_source.json:dimm_local",
                  mechanical_acceptance=False,thresholds="DEMONSTRATION_NOT_MATERIAL_LIMITS")
        # Check all sensor streams before creating an episode.
        self.observe(save_images=False,require_images=True)
        self.active=Episode(ROOT/"outputs/datasets/task1_dimm_insert/raw",meta)
        self.active.path.joinpath("initial_session.usda").write_text(self.stage.GetSessionLayer().ExportToString())
        self.pending_obs=None;self.clock=0.;self.subcommands=[]
        self.action=keyboard_action(self.held,self.cfg["action"]) if self.enabled else np.zeros(7)
        self.notice="RECORDING "+self.active.path.name
    def finish(self,outcome,reason=""):
        if not self.active:return
        if outcome=="failed":
            path=str(self.active.path)
            load("_task1_discard",TASK/"teleop/discard_recording.py").discard_current(self)
            self.record_error="";self.last_saved_path=""
            self.record_event("discarded",path=path)
            return
        # A partial control interval is preserved as a diagnostic, not a transition.
        if self.clock>0:self.active.physics({"partial_interval_s":self.clock,"action":self.action,"reason":"episode_end"})
        checks={"mechanical_success":None,"safety_pass":False if self.violations else None,
                "unverified":["hook/notch engagement","release stability criteria","material limits","full task history"],
                "violations":sorted(self.violations),"sensor_errors":self.sensor_errors}
        path=self.active.finish(outcome,reason,checks);self.active=None;self.pending_obs=None
        self.last_saved_path=str(path)
        self.record_event("saved",path=str(path),outcome=outcome,reason=reason)
        from tasks.task1_dimm_insert.env.validate_recording import validate
        try:
            result=validate(path,write=True)
            self.notice=f"Saved {path.name} | data_valid={result['data_valid']} | success pending mechanical review"
        except Exception as e:
            self.notice="SAVED but validation failed: "+str(path)+" | "+repr(e)
            self.record_event("validation_error",path=str(path),reason=repr(e))
    def estop(self):
        self.finish("interrupted","user_estop");self.enabled=False;self.held.clear();self.action[:]=0
        if self.controller.ready:self.controller.stop("USER E-STOP")
        self.timeline.pause();self.notice="E-STOP. Recording closed."
    def reset(self):
        import asyncio,omni.usd
        self.finish("interrupted","reset");self.close()
        async def reopen():
            import omni.kit.app
            self.timeline.stop()
            await omni.usd.get_context().open_stage_async(str(TASK/"scene/collection.usda"))
            import builtins
            import omni.kit.app
            for _ in range(3):await omni.kit.app.get_app().next_update_async()
            builtins._task1_collection=CollectionPanel()
        asyncio.ensure_future(reopen())
    def timeline_event(self,event):
        import omni.timeline
        if event.type in (int(omni.timeline.TimelineEventType.PAUSE),int(omni.timeline.TimelineEventType.STOP)):
            self.finish("interrupted","timeline_pause_or_stop");self.enabled=False;self.held.clear()
            if event.type==int(omni.timeline.TimelineEventType.STOP):
                self.controller.ready=False;self.bodies={};self.t=0.;self.history.clear();self.image_cache={}
    def on_input(self,event):
        import carb.input
        if self.closed:return True
        if event.deviceType!=carb.input.DeviceType.KEYBOARD:return True
        e=event.event
        name=e.input if isinstance(e.input,str) else str(getattr(e.input,"name",e.input))
        keys={"UP","DOWN","LEFT","RIGHT","PAGE_UP","PAGE_DOWN","LEFT_SHIFT","RIGHT_SHIFT",
              "LEFT_CONTROL","RIGHT_CONTROL","LEFT_BRACKET","RIGHT_BRACKET","ESCAPE"}
        if name not in keys:return True
        if name=="ESCAPE" and e.type==carb.input.KeyboardEventType.KEY_PRESS:self.estop();return False
        if not self.enabled:return True
        if e.type in (carb.input.KeyboardEventType.KEY_PRESS,carb.input.KeyboardEventType.KEY_REPEAT):self.held.add(name)
        elif e.type==carb.input.KeyboardEventType.KEY_RELEASE:self.held.discard(name)
        return False
    def on_contacts(self,headers,data):
        from pxr import PhysicsSchemaTools
        for h in headers:
            a=str(PhysicsSchemaTools.intToSdfPath(h.collider0));b=str(PhysicsSchemaTools.intToSdfPath(h.collider1))
            if not any(p in a+b for p in ["/World/Robot","/World/MatchedDIMM","/World/MatchedSocket"]):continue
            for i in range(h.contact_data_offset,h.contact_data_offset+h.num_contact_data):
                c=data[i];self.pending_contacts.append(dict(a=a,b=b,point=list(c.position),normal=list(c.normal),
                    impulse=list(c.impulse),separation_m=float(c.separation)))
    def measure_contacts(self,dt):
        contacts=self.pending_contacts;self.pending_contacts=[]
        metrics={"left_press_N":0.,"right_press_N":0.,"dimm_environment_N":0.,"finger_left_N":0.,"finger_right_N":0.}
        for c in contacts:
            # PhysX normal impulse / actual physics interval; excludes friction.
            c["force_N"]=(np.asarray(c["impulse"])/dt).tolist()
            mag=float(abs(np.dot(c["force_N"],c["normal"])));c["normal_force_N"]=mag
            both=c["a"]+" "+c["b"]
            for side in ["left","right"]:
                if side.title()+"Latch" in both and "/World/Robot/" in both:metrics[side+"_press_N"]+=mag
            if "/World/MatchedDIMM" in both:
                if "panda_leftfinger" in both:metrics["finger_left_N"]+=mag
                elif "panda_rightfinger" in both:metrics["finger_right_N"]+=mag
                else:metrics["dimm_environment_N"]+=mag
        for k,v in metrics.items():
            self.interval_peaks[k]=max(self.interval_peaks.get(k,0.),v)
            limit=self.cfg["demonstration_limits"].get(k)
            if limit is not None and v>limit:self.violations.add(k)
        self.contact_metrics=metrics
        if getattr(self,"feedback",None):self.feedback.consume(contacts)
        return contacts
    def task_metrics(self,poses):
        from pxr import UsdGeom,UsdPhysics,Gf
        cache=UsdGeom.XformCache()
        socket=self.stage.GetPrimAtPath("/World/MatchedSocket/Asset")
        sm=cache.GetLocalToWorldTransform(socket)
        local=sm.GetInverse().Transform(Gf.Vec3d(*[float(v) for v in poses["dimm"][:3]]))
        result={"dimm_position_in_socket_m":list(local),"hook_engagement":None}
        for side in ["left","right"]:
            joint=UsdPhysics.RevoluteJoint(self.stage.GetPrimAtPath("/World/MatchedSocket/Asset/Joints/"+side.title()+"Hinge"))
            if not joint:result[side+"_angle_deg"]=None;continue
            b0=self.stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])
            m=cache.GetLocalToWorldTransform(b0);q=m.ExtractRotationQuat().GetNormalized()
            def rotation(q):return Rotation.from_quat([*q.GetImaginary(),q.GetReal()])
            r0=rotation(q)*rotation(joint.GetLocalRot0Attr().Get())
            r1=Rotation.from_quat(poses[side][3:])*rotation(joint.GetLocalRot1Attr().Get())
            axis={"X":0,"Y":1,"Z":2}[joint.GetAxisAttr().Get()]
            result[side+"_angle_deg"]=float(np.rad2deg((r0.inv()*r1).as_rotvec()[axis]))
        return result
    def observe(self,save_images=True,require_images=False):
        if not self.controller.ready:raise RuntimeError("Controller not initialized")
        q=self.controller.get_joint_positions();qd=self.controller.view.get_dof_velocities()[0]
        hand=self.hand.get_transforms()[0].copy()
        poses={k:v.get_transforms()[0].copy() for k,v in self.bodies.items()}
        velocities={k:v.get_velocities()[0].copy() for k,v in self.bodies.items()}
        obs=dict(sim_time=self.t,joint_positions=q,joint_velocities=qd,ee_position_m=hand[:3],
                 ee_quaternion_xyzw=hand[3:],gripper_width_m=float(q[self.controller.fingers].sum()),
                 task_truth=dict(poses=poses,velocities=velocities,metrics=self.task_metrics(poses),assistance=self.assistance_state),
                 controller_state=dict(goal_position=self.controller.goal[0],goal_quaternion_xyzw=self.controller.goal[1].as_quat(),
                    gripper_target=self.controller.width_goal,armed=self.controller.armed),
                 images={},sensor_valid=True)
        for key in self.cfg["required_cameras"]:
            item=self.image_cache.get(key)
            if item is None or self.t-item["sim_time"]>self.cfg["max_image_age_s"] or item["sim_time"]>self.t+1/self.cfg["physics_hz"]:
                if require_images:raise RuntimeError("No fresh rendered image: "+key)
                obs["sensor_valid"]=False;continue
            from tasks.task1_dimm_insert.env.image_quality import inspect_rgb
            health=inspect_rgb(item["rgb"])
            if not health["valid"]:
                if require_images:raise RuntimeError("Camera image rejected: "+key+" "+health["reason"])
                obs["sensor_valid"]=False
            meta={k:v for k,v in item.items() if k!="rgb"}
            meta["quality"]=health
            if self.active and save_images:
                name=f"images/{key}_{item['frame']:09d}.png"
                if name not in self.saved_frames:
                    from PIL import Image
                    Image.fromarray(item["rgb"]).save(self.active.path/name);self.saved_frames[name]=True
                meta["file"]=name
            obs["images"][key]=meta
        return clean(obs)
    def physics_step(self,dt):
        if self.closed or dt<=0:return
        self.t+=dt
        try:
            contacts=self.measure_contacts(dt)
            if not self.controller.ready:return
            if self.assistance:
                poses={k:v.get_transforms()[0].copy() for k,v in self.bodies.items()}
                measured=self.task_metrics(poses)
                for side in ["left","right"]:
                    angle=measured.get(side+"_angle_deg")
                    measured[side+"_angle_abs_deg"]=None if angle is None else abs(angle)
                measured.update(self.contact_metrics)
                self.assistance_state=self.assistance.update_latch_soft_hold(self.stage,self.safety,measured,self.calib)
            speed=max([float(np.linalg.norm(v.get_velocities()[0,:3])) for v in self.bodies.values()]+[0.])
            self.history.append((self.t,speed))
            if self.active:
                self.active.physics(dict(sim_time=self.t,dt=dt,contacts=contacts,metrics=self.contact_metrics,
                    joint_drive_targets=self.controller.view.get_dof_position_targets()[0],
                    controller_error=self.controller.error,assistance=self.assistance_state))
            if self.active:
                if self.pending_obs is None:
                    self.pending_obs=self.observe(require_images=True)
                    self.clock=0.
                else:
                    self.clock+=dt
                if self.clock+1e-9>=1/self.cfg["control_hz"]:
                    nxt=self.observe()
                    self.active.append(self.pending_obs,self.action,nxt,
                        dict(contact_peaks=self.interval_peaks,controller_commands=self.subcommands,
                             controller_error=self.controller.error,operator="human_keyboard"))
                    self.last_write_wall=time.monotonic()
                    self.pending_obs=nxt;self.clock=0.;self.subcommands=[];self.interval_peaks={}
                    self.action=keyboard_action(self.held,self.cfg["action"]) if self.enabled else np.zeros(7)
            else:self.action=keyboard_action(self.held,self.cfg["action"]) if self.enabled else np.zeros(7)
            if not self.enabled:return
            command=apply_velocity(self.controller,self.action,dt,self.cfg["action"])
            # Soft XY attract near socket entrance (assist only; never auto-success).
            try:
                poses_a={k:v.get_transforms()[0].copy() for k,v in self.bodies.items()} if self.bodies else {}
                metrics_a=self.task_metrics(poses_a) if poses_a else {}
                self.soft_attract_state=apply_soft_xy_attract(
                    self.controller,self.stage,self.calib,self.cfg.get("teleop_assist"),dt,metrics_a)
                command=dict(command,soft_xy_attract=self.soft_attract_state)
            except Exception as e:
                self.soft_attract_state={"active":False,"reason":"error:"+repr(e)}
            if self.active:self.subcommands.append(dict(time=self.t,**command))
            self.controller.tick(dt)
            if self.controller.error:
                self.violations.add("controller_fault")
                if not self.controller.armed:
                    self.finish("interrupted","controller_disarmed:"+str(self.controller.error));self.enabled=False
        except Exception as e:
            self.sensor_errors.append(repr(e));self.notice="Collection error: "+repr(e)
            self.finish("interrupted",repr(e));self.enabled=False
            if self.controller.ready:self.controller.stop("COLLECTION ERROR")
    def update(self,event):
        if self.closed:return
        if getattr(self,"feedback",None):self.feedback.update()
        import omni.usd
        if omni.usd.get_context().get_stage()!=self.stage:self.close();return
        for key,cam in self.cameras.items():
            try:
                frame=cam.get_current_frame();rgb=frame.get("rgb",frame.get("rgba"))
                native=frame.get("rendering_frame",0);ts=float(frame.get("rendering_time",0.))
                token=json.dumps(native,sort_keys=True)
                if rgb is not None and np.asarray(rgb).size and ts>0:
                    previous=self.image_cache.get(key,{})
                    if token!=previous.get("native_frame"):
                        self.image_cache[key]={"frame":int(previous.get("frame",0))+1,"native_frame":token,
                            "sim_time":ts,"rgb":np.asarray(rgb,dtype=np.uint8)[:,:,:3].copy()}
            except Exception as e:self.camera_errors.append(repr(e))
        if self.ui and not self.ui.visible:self.close();return
        if self.ui and time.monotonic()-self.last_refresh>.1:
            self.last_refresh=time.monotonic()
            banner,color=self.recording_status()
            self.record_banner.text=banner
            self.record_banner.style={"color":color,"font_size":20}
            self.status.text=self.notice
            metrics=getattr(self,"contact_metrics",{})
            sa=getattr(self,"soft_attract_state",{}) or {}
            align=sa.get("align_xy_m"); dz=sa.get("dz_from_entry_m")
            guide=(" | align_xy="+("{:.1f}mm".format(align*1000) if align is not None else "?")+
                   " dz="+("{:.1f}mm".format(dz*1000) if dz is not None else "?")+
                   (" SNAP" if sa.get("active") else ""))
            self.values.text="Contact N: "+str({k:round(v,3) for k,v in metrics.items()})+"\nKeys: "+",".join(sorted(self.held))+guide
            ratios=[v/self.cfg["demonstration_limits"][k] for k,v in metrics.items() if k in self.cfg["demonstration_limits"]]
            color=0xff4040ff if self.violations else (0xff40ccff if max(ratios,default=0)>=.7 else 0xff60cc60)
            self.values.style={"color":color}
            self.quality.text=f"Images {len(self.image_cache)}/3 | records {self.active.count if self.active else 0} | mechanical_success=UNVERIFIED"
    def close(self):
        if self.closed:return
        self.finish("interrupted","panel_closed");self.enabled=False;self.held.clear()
        if self.controller.ready:self.controller.stop("COLLECTION CLOSED")
        self.closed=True
        if getattr(self,"feedback",None):self.feedback.close()
        if hasattr(self,"preview_module"):self.preview_module.close(self)
        self.input.unsubscribe_to_input_events(self.key_sub)
        self.physics_sub=None;self.contact_sub=None;self.update_sub=None;self.timeline_sub=None
        if self.ui:self.ui.visible=False
        for cam in self.cameras.values():
            try:cam.pause()
            except Exception:pass
