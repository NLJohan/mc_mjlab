#include "hpp/controller_instance.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <mc_control/MCController.h>
#include <stdexcept>
#include "hpp/instance_datastore_plugin.hpp"
#include "hpp/utils.hpp"

ControllerInstance::ControllerInstance(const GlobalConfiguration &configuration, const IoLayout &layout)
    : m_configuration(configuration), m_layout(layout), m_initialized(false), m_failed(false)
{}

void ControllerInstance::initialize(IoInput input, IoOutput output)
{
    if (m_initialized) throw std::logic_error("controller is already initialized");

    m_q.assign(m_layout.input.joint_order.size(), 0.0);
    m_qd.assign(m_q.size(), 0.0);
    m_tau.assign(m_q.size(), 0.0);

    m_controller = std::make_unique<mc_control::MCGlobalController>(m_configuration);

    std::vector<std::string> from = m_layout.input.joint_order;
    std::vector<std::string> to;

    for (const auto &joint : m_controller->robot().mb().joints())
    {
        to.push_back(joint.name());
    }

    m_io_to_mbc.clear();

    for (const auto &name : from)
    {
        const auto match = std::find(to.begin(), to.end(), name);
        m_io_to_mbc.push_back(match == to.end() ? invalid_index : static_cast<std::size_t>(match - to.begin()));
    }

    const auto pose = prepare_reset(input);

    // Seed encoder values before init(), since MCGlobalController::init() runs
    // resetObserverPipelines() internally and EncoderObserver requires non-empty
    // encoderValues() on that first pipeline run.
    io_to_encoders(input.subspan(m_layout.input.q_offset()), m_q);
    m_controller->setEncoderValues(m_q);

    m_controller->init(m_q, pose);

    finish_reset(input, pose);
    apply_output(output);

    m_failed      = false;
    m_initialized = true;
}

void ControllerInstance::reset(IoInput input, IoOutput output)
{
    if (!m_initialized)
    {
        initialize(input, output);
        return;
    }

    const auto pose = prepare_reset(input);

    // Soft reset, matching the old mc_rtc_controller_host.py reset_envs()'s
    // init() branch: re-run MCGlobalController::init() on the SAME, still-alive
    // controller instead of the destructive reset()/erase()+AddController()
    // path. That binding never exposed MCGlobalController::reset() at all, so
    // in production every past reset already took exactly this path.
    const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
    auto &controller = m_controller->controller();

    // Zero velocity/acceleration explicitly -- init() does not touch these,
    // and leaving stale values here produced garbage comVelocity() readings
    // post-reset (observer computing an innovation against stale sensor
    // state), per the original bug this sequence was written to fix.
    if (!m_layout.input.body_sensors.empty() &&
        std::find(m_layout.input.body_sensors.begin(), m_layout.input.body_sensors.end(),
                  InputLayout::floating_base_sensor) != m_layout.input.body_sensors.end())
    {
        // FloatingBase gets position/orientation set here too, from the same
        // fresh pose used for init()'s attitude below -- without this the
        // sensor buffer still holds the previous episode's last pre-fall pose
        // at the moment init() runs the observer pipeline.
        m_controller->setSensorPosition(pose.translation());
        m_controller->setSensorOrientation(Eigen::Quaterniond(pose.rotation()));
        m_controller->setSensorLinearVelocity(zero);
        m_controller->setSensorAngularVelocity(zero);
        m_controller->setSensorLinearAcceleration(zero);
    }
    m_controller->setSensorAngularVelocity(zero);
    m_controller->setSensorLinearAcceleration(zero);

    io_to_encoders(input.subspan(m_layout.input.q_offset()), m_q);
    m_controller->setEncoderValues(m_q);

    m_controller->init(m_q, pose);

    finish_reset(input, pose);
    apply_output(output);

    m_failed = false;
}

sva::PTransformd ControllerInstance::prepare_reset(IoInput input)
{
    io_to_encoders(input.subspan(m_layout.input.q_offset()), m_q);

    const auto root            = input.subspan(m_layout.input.root_offset(), InputLayout::root_state_size);
    const auto world_from_body = utils::geometry::quaternion_xyzw(root.subspan(3, 4));
    const auto body_from_world = world_from_body.inverse();
    return sva::PTransformd(body_from_world, utils::geometry::vector3(root, 0));
}

void ControllerInstance::finish_reset(IoInput input, const sva::PTransformd &pose)
{
    auto &controller = m_controller->controller();
    controller.realRobot().posW(pose);

    apply_input(input);

    controller.reset({controller.robot().mbc().q});

    instance_datastore_plugin::register_entries(controller, m_layout);

    utils::datastore::validate<double>(controller.datastore(), m_layout.input.datastore_scalar, true);
    utils::datastore::validate<Eigen::Vector3d>(controller.datastore(), m_layout.input.datastore_vector3, true);
    utils::datastore::validate<double>(controller.datastore(), m_layout.output.datastore_scalar, false);
    utils::datastore::validate<Eigen::Vector3d>(controller.datastore(), m_layout.output.datastore_vector3, false);

    m_controller->running = true;
}

OutputLayout::ControllerStatus ControllerInstance::step(IoInput input, IoOutput output)
{
    if (m_failed || !m_initialized)
    {
        return m_failed ? OutputLayout::QP_FAILED : OutputLayout::WORKER_FAILED;
    }

    apply_input(input);

    auto &datastore = m_controller->controller().datastore();

    for (std::size_t i = 0; i < m_layout.input.datastore_scalar.size(); ++i)
    {
        utils::datastore::write<double>(
            datastore, m_layout.input.datastore_scalar[i], input[m_layout.input.datastore_scalar_offset() + i]);
    }

    for (std::size_t i = 0; i < m_layout.input.datastore_vector3.size(); ++i)
    {
        utils::datastore::write<Eigen::Vector3d>(
            datastore,
            m_layout.input.datastore_vector3[i],
            utils::geometry::vector3(input, m_layout.input.datastore_vector3_offset() + 3 * i));
    }

    m_failed = !m_controller->run();

    if (!m_failed)
    {
        apply_output(output);
    }

    return m_failed ? OutputLayout::QP_FAILED : OutputLayout::OK;
}

void ControllerInstance::apply_input(IoInput input)
{
    io_to_encoders(input.subspan(m_layout.input.q_offset()), m_q);
    io_to_encoders(input.subspan(m_layout.input.qd_offset()), m_qd);
    io_to_encoders(input.subspan(m_layout.input.tau_offset()), m_tau);

    m_controller->setEncoderValues(m_q);
    m_controller->setEncoderVelocities(m_qd);
    m_controller->setJointTorques(m_tau);

    const auto  root            = input.subspan(m_layout.input.root_offset(), InputLayout::root_state_size);
    const auto  world_from_body = utils::geometry::quaternion_xyzw(root.subspan(3, 4));
    const auto  body_from_world = world_from_body.inverse();
    const auto &base            = InputLayout::floating_base_sensor;

    m_controller->setSensorPositions({{base, utils::geometry::vector3(root, 0)}});
    m_controller->setSensorOrientations({{base, body_from_world}});
    m_controller->setSensorLinearVelocities({{base, utils::geometry::vector3(root, 7)}});

    std::map<std::string, Eigen::Vector3d> gyros, accelerometers;

    for (std::size_t i = 0; i < m_layout.input.body_sensors.size(); ++i)
    {
        const auto &name     = m_layout.input.body_sensors[i];
        const auto  offset   = m_layout.input.body_sensors_offset() + InputLayout::body_sensor_size * i;
        gyros[name]          = utils::geometry::vector3(input, offset);
        accelerometers[name] = utils::geometry::vector3(input, offset + 3);
    }

    m_controller->setSensorAngularVelocities(gyros);
    m_controller->setSensorLinearAccelerations(accelerometers);

    std::map<std::string, sva::ForceVecd> wrenches;

    for (std::size_t i = 0; i < m_layout.input.force_sensors.size(); ++i)
    {
        const auto offset = m_layout.input.force_sensors_offset() + InputLayout::force_sensor_size * i;
        // MuJoCo measures force on the site; mc_rtc expects the reaction on the robot.
        wrenches.emplace(
            m_layout.input.force_sensors[i],
            sva::ForceVecd(-utils::geometry::vector3(input, offset + 3), -utils::geometry::vector3(input, offset)));
    }

    m_controller->setWrenches(wrenches);
}

void ControllerInstance::apply_output(IoOutput output)
{
    const auto &mbc = m_controller->robot().mbc();

    const auto count = m_layout.output.joint_order.size();
    mbc_to_io(output.subspan(m_layout.output.q_offset(), count), mbc.q);
    mbc_to_io(output.subspan(m_layout.output.qd_offset(), count), mbc.alpha);
    mbc_to_io(output.subspan(m_layout.output.tau_offset(), count), mbc.jointTorque);

    const auto &datastore = m_controller->controller().datastore();

    for (std::size_t i = 0; i < m_layout.output.datastore_scalar.size(); ++i)
    {
        output[m_layout.output.datastore_scalar_offset() + i] =
            utils::datastore::read<double>(datastore, m_layout.output.datastore_scalar[i]);
    }

    for (std::size_t i = 0; i < m_layout.output.datastore_vector3.size(); ++i)
    {
        Eigen::Map<Eigen::Vector3d> target(output.data() + m_layout.output.datastore_vector3_offset() + 3 * i);
        target = utils::datastore::read<Eigen::Vector3d>(datastore, m_layout.output.datastore_vector3[i]);
    }
}

void ControllerInstance::io_to_encoders(IoInput io, std::vector<double> &encoders)
{
    std::copy_n(io.begin(), encoders.size(), encoders.begin());
}

void ControllerInstance::mbc_to_io(IoOutput io, const std::vector<std::vector<double>> &mbc)
{
    std::fill(io.begin(), io.end(), 0.0);

    for (std::size_t i = 0; i < m_io_to_mbc.size(); ++i)
    {
        const auto source = m_io_to_mbc[i];

        if (source != invalid_index && mbc[source].size() == 1)
        {
            io[i] = mbc[source][0];
        }
    }
}

std::vector<std::pair<std::string, std::string>> ControllerInstance::get_available_datastore_entries()
{
    std::vector<std::pair<std::string, std::string>> entries;

    const auto &datastore = m_controller->controller().datastore();

    for (const auto &key : datastore.keys())
    {
        entries.emplace_back(key, datastore.type(key));
    }

    return entries;
}

bool ControllerInstance::failed() const
{
    return m_failed;
}

bool ControllerInstance::initialized() const
{
    return m_initialized;
}
