#pragma once

#include <Eigen/Geometry>
#include <boost/interprocess/mapped_region.hpp>
#include <cstddef>
#include <functional>
#include <limits>
#include <mc_rtc/DataStore.h>
#include <stdexcept>
#include <string>
#include <vector>

struct WorkerStartMessage;

namespace utils::compat
{
inline constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();
template <typename T, std::size_t Extent = dynamic_extent> class span
{
public:
    span() noexcept : data_(nullptr), size_(0) {}
    span(T *data, std::size_t size) noexcept : data_(data), size_(size) {}

    template <typename Container> span(Container &c) : data_(c.data()), size_(c.size()) {}

    template <std::size_t N> span(T (&arr)[N]) noexcept : data_(arr), size_(N) {}

    T &operator[](std::size_t i) const { return data_[i]; }
    T *data() const noexcept { return data_; }std::size_t size() const noexcept { return size_; }
    std::size_t size_bytes() const noexcept { return size_ * sizeof(T); }
    bool empty() const noexcept { return size_ == 0; }

    T *begin() const noexcept { return data_; }
    T *end() const noexcept { return data_ + size_; }

    span<T> first(std::size_t count) const { return span<T>(data_, count); }

    span<T> subspan(std::size_t offset, std::size_t count = dynamic_extent) const
    {
        std::size_t n = (count == dynamic_extent) ? (size_ - offset) : count;
        return span<T>(data_ + offset, n);
    }

private:
    T *data_;
    std::size_t size_;
};
} // namespace utils::compat

namespace utils::geometry
{
    inline Eigen::Map<const Eigen::Vector3d> vector3(utils::compat::span<const double> row, std::size_t offset)
    {
        return Eigen::Map<const Eigen::Vector3d>(row.data() + offset);
    }

    inline Eigen::Quaterniond quaternion_xyzw(utils::compat::span<const double> values)
    {
        return Eigen::Quaterniond(values[3], values[0], values[1], values[2]);
    }
} // namespace utils::geometry

namespace utils::datastore
{
    template <typename T> T read(const mc_rtc::DataStore &datastore, const std::string &name)
    {
        if (datastore.type(name) == mc_rtc::type_name<std::function<const T &()>>())
            return datastore.call<const T &>(name);
        return datastore.call<T>(name);
    }

    template <typename T> void write(const mc_rtc::DataStore &datastore, const std::string &name, const T &value)
    {
        if (datastore.type(name) == mc_rtc::type_name<std::function<void(const T &)>>())
            datastore.get<std::function<void(const T &)>>(name)(value);
        else
            datastore.get<std::function<void(T)>>(name)(value);
    }

    template <typename T>
    void validate(const mc_rtc::DataStore &datastore, const std::vector<std::string> &names, bool input)
    {
        const auto &value_type =
            input ? mc_rtc::type_name<std::function<void(T)>>() : mc_rtc::type_name<std::function<T()>>();
        const auto &reference_type = input ? mc_rtc::type_name<std::function<void(const T &)>>()
                                           : mc_rtc::type_name<std::function<const T &()>>();
        for (const auto &name : names)
        {
            const auto type = datastore.type(name);
            if (type != value_type && type != reference_type)
                throw std::invalid_argument("unsupported datastore callback: " + name + " (" + type + ")");
        }
    }
} // namespace utils::datastore

namespace utils::shared_memory
{
    struct WorkerIoMappings
    {
            boost::interprocess::mapped_region input;
            boost::interprocess::mapped_region output;
    };

    [[nodiscard]] WorkerIoMappings
        map_worker_io(const WorkerStartMessage &message, utils::compat::span<const double> &input, utils::compat::span<double> &output);
} // namespace utils::shared_memory
